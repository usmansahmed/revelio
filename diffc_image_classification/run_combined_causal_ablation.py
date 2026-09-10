import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binomtest, wilcoxon
from torch.utils.data import DataLoader, Subset


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_tensor(path, device="cpu"):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing tensor file: {path}")

    return torch.load(path, map_location=device)


def load_feature_stats(feature_dir, device):
    feature_dir = Path(feature_dir)

    return {
        "label_purity": load_tensor(
            feature_dir / "label_purity_top10.pt", device
        ),
        "majority_label": load_tensor(
            feature_dir / "majority_label_top10.pt", device
        ),
        "valid_count": load_tensor(
            feature_dir / "valid_top_count_top10.pt", device
        ),
        "mean_acts": load_tensor(
            feature_dir / "sae_mean_acts.pt", device
        ),
        "sparsity": load_tensor(
            feature_dir / "sae_sparsity.pt", device
        ),
    }


def load_ksae(checkpoint_path, device, default_k=32):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["state_dict"]

    if "cfg" in checkpoint and hasattr(checkpoint["cfg"], "k"):
        k = checkpoint["cfg"].k
    else:
        k = default_k

    return {
        "W_enc": state["W_enc"].to(device),
        "b_enc": state["b_enc"].to(device),
        "W_dec": state["W_dec"].to(device),
        "b_dec": state["b_dec"].to(device),
        "k": k,
        "n_features": state["W_enc"].shape[1],
    }


def ksae_encode(x, ksae):
    # Match the trained k-SAE exactly: TopK directly, no ReLU.
    pre_acts = (
        (x - ksae["b_dec"]) @ ksae["W_enc"]
        + ksae["b_enc"]
    )

    top_values, top_indices = torch.topk(
        pre_acts,
        k=ksae["k"],
        dim=-1,
    )

    sparse_acts = torch.zeros_like(pre_acts)

    sparse_acts.scatter_(
        dim=-1,
        index=top_indices,
        src=top_values,
    )

    return sparse_acts


def ksae_decode(sparse_acts, ksae):
    return sparse_acts @ ksae["W_dec"] + ksae["b_dec"]


def select_target_features(
    summary_df,
    target_class,
    min_purity,
    min_valid,
):
    selected = summary_df[
        (summary_df["majority_label"] == target_class)
        & (summary_df["label_purity"] >= min_purity)
        & (summary_df["valid_top_count"] >= min_valid)
    ]

    return selected["feature_id"].astype(int).tolist()


def score_active_candidates(
    active_ids,
    sparse_row,
    stats,
    ranking_method,
):
    purity = stats["label_purity"][active_ids]
    mean_acts = stats["mean_acts"][active_ids]
    sparsity = stats["sparsity"][active_ids]
    actual_activation = sparse_row[active_ids]

    if ranking_method == "actual_activation":
        return actual_activation

    if ranking_method == "purity_actual_activation":
        return purity * actual_activation

    if ranking_method == "purity_mean_activation":
        return purity * mean_acts

    if ranking_method == "sparse_class_specific":
        return purity * mean_acts * sparsity

    if ranking_method == "purity":
        return purity

    if ranking_method == "sparsity":
        return sparsity

    raise ValueError(
        f"Unknown active_feature_ranking: {ranking_method}"
    )


def select_active_class_features(
    sparse_row,
    stats,
    target_class,
    min_purity,
    min_valid,
    max_ablate_per_image,
    ranking_method,
    target_feature_ids,
):
    active_ids = torch.nonzero(
        sparse_row != 0,
        as_tuple=False,
    ).flatten()

    if active_ids.numel() == 0:
        return active_ids, active_ids, active_ids

    allowed_ids = torch.tensor(
        target_feature_ids,
        device=active_ids.device,
        dtype=active_ids.dtype,
    )

    class_mask = (
        torch.isin(active_ids, allowed_ids)
        & (
            stats["majority_label"][active_ids]
            == target_class
        )
        & (
            stats["label_purity"][active_ids]
            >= min_purity
        )
        & (
            stats["valid_count"][active_ids]
            >= min_valid
        )
    )

    target_ids = active_ids[class_mask]

    if target_ids.numel() > max_ablate_per_image:
        scores = score_active_candidates(
            target_ids,
            sparse_row,
            stats,
            ranking_method,
        )

        order = torch.argsort(
            scores,
            descending=True,
        )

        target_ids = target_ids[
            order[:max_ablate_per_image]
        ]

    # Active non-target-class neurons are used as controls.
    # They are matched to targeted neurons by absolute
    # activation magnitude without replacement.
    control_pool = active_ids[
        stats["majority_label"][active_ids]
        != target_class
    ]

    matched_random_ids = []
    available = control_pool

    for target_id in target_ids:
        if available.numel() == 0:
            break

        distances = (
            sparse_row[available].abs()
            - sparse_row[target_id].abs()
        ).abs()

        # Tiny random jitter only affects exact ties.
        distances = (
            distances
            + torch.rand_like(distances) * 1e-12
        )

        match_position = torch.argmin(distances)

        matched_random_ids.append(
            available[match_position]
        )

        available = torch.cat(
            (
                available[:match_position],
                available[match_position + 1:],
            )
        )

    if matched_random_ids:
        random_ids = torch.stack(
            matched_random_ids
        )
    else:
        random_ids = control_pool[:0]

    return active_ids, target_ids, random_ids


def build_feature_variants(
    features,
    ksae,
    stats,
    class_cfg,
    feature_counter,
):
    pooled = features.mean(dim=(2, 3))
    sparse_acts = ksae_encode(pooled, ksae)

    sparse_recon = sparse_acts
    sparse_target = sparse_acts.clone()
    sparse_random = sparse_acts.clone()

    batch_interventions = []

    for i in range(sparse_acts.size(0)):
        (
            active_ids,
            target_ids,
            random_ids,
        ) = select_active_class_features(
            sparse_acts[i],
            stats,
            target_class=class_cfg["target_class"],
            min_purity=class_cfg["min_purity"],
            min_valid=class_cfg["min_valid"],
            max_ablate_per_image=(
                class_cfg["max_ablate_per_image"]
            ),
            ranking_method=(
                class_cfg["active_feature_ranking"]
            ),
            target_feature_ids=(
                class_cfg["target_feature_ids"]
            ),
        )

        if target_ids.numel() > 0:
            target_activation_sum = float(
                sparse_acts[
                    i, target_ids
                ].sum().item()
            )
        else:
            target_activation_sum = 0.0

        if random_ids.numel() > 0:
            random_activation_sum = float(
                sparse_acts[
                    i, random_ids
                ].sum().item()
            )
        else:
            random_activation_sum = 0.0

        if target_ids.numel() > 0:
            sparse_target[i, target_ids] = 0.0

        if random_ids.numel() > 0:
            sparse_random[i, random_ids] = 0.0

        batch_interventions.append({
            "target_neuron_ids": [
                int(x)
                for x in target_ids
                .detach()
                .cpu()
                .tolist()
            ],
            "random_neuron_ids": [
                int(x)
                for x in random_ids
                .detach()
                .cpu()
                .tolist()
            ],
            "number_target_neurons_ablated":
                int(target_ids.numel()),
            "number_random_neurons_ablated":
                int(random_ids.numel()),
            "sum_target_activation":
                target_activation_sum,
            "sum_random_activation":
                random_activation_sum,
        })

        for fid in (
            target_ids.detach().cpu().tolist()
        ):
            fid = int(fid)

            feature_counter[
                "target_feature_counts"
            ][fid] = (
                feature_counter[
                    "target_feature_counts"
                ].get(fid, 0)
                + 1
            )

            feature_counter[
                "target_feature_activation_sums"
            ][fid] = (
                feature_counter[
                    "target_feature_activation_sums"
                ].get(fid, 0.0)
                + float(
                    sparse_acts[
                        i, fid
                    ].item()
                )
            )

        for fid in (
            random_ids.detach().cpu().tolist()
        ):
            fid = int(fid)

            feature_counter[
                "random_feature_counts"
            ][fid] = (
                feature_counter[
                    "random_feature_counts"
                ].get(fid, 0)
                + 1
            )

        feature_counter[
            "total_active_features"
        ] += int(active_ids.numel())

        feature_counter[
            "total_target_ablated"
        ] += int(target_ids.numel())

        feature_counter[
            "total_random_ablated"
        ] += int(random_ids.numel())

        if target_ids.numel() > 0:
            feature_counter[
                "images_with_target_features"
            ] += 1

    recon_pooled = ksae_decode(
        sparse_recon,
        ksae,
    )

    target_pooled = ksae_decode(
        sparse_target,
        ksae,
    )

    random_pooled = ksae_decode(
        sparse_random,
        ksae,
    )

    # Reconstruction baseline.
    recon_features = (
        features
        + (recon_pooled - pooled)[
            :, :, None, None
        ]
    )

    # Residual-preserving targeted intervention.
    target_features = (
        features
        + (target_pooled - recon_pooled)[
            :, :, None, None
        ]
    )

    # Residual-preserving random intervention.
    random_features = (
        features
        + (random_pooled - recon_pooled)[
            :, :, None, None
        ]
    )

    return (
        recon_features,
        target_features,
        random_features,
        batch_interventions,
    )


def empty_metrics():
    return {
        "loss_sum": 0.0,
        "correct": 0,
        "total": 0,
        "true_conf_sum": 0.0,
        "target_conf_sum": 0.0,
    }


def update_metrics(
    metrics,
    logits,
    labels,
    target_class,
):
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

    metrics["loss_sum"] += F.cross_entropy(
        logits,
        labels,
        reduction="sum",
    ).item()

    metrics["correct"] += (
        preds == labels
    ).sum().item()

    metrics["total"] += labels.numel()

    metrics["true_conf_sum"] += probs[
        torch.arange(
            labels.numel(),
            device=labels.device,
        ),
        labels,
    ].sum().item()

    metrics["target_conf_sum"] += probs[
        :, target_class
    ].sum().item()


def finalize_metrics(metrics):
    total = metrics["total"]

    if total == 0:
        return {
            "accuracy": 0.0,
            "cross_entropy": 0.0,
            "true_class_confidence": 0.0,
            "target_class_confidence": 0.0,
            "total": 0,
        }

    return {
        "accuracy":
            metrics["correct"] / total,
        "cross_entropy":
            metrics["loss_sum"] / total,
        "true_class_confidence":
            metrics["true_conf_sum"] / total,
        "target_class_confidence":
            metrics["target_conf_sum"] / total,
        "total":
            total,
    }


def sorted_count_dict(
    count_dict,
    top_n=50,
):
    rows = [
        {
            "feature_id": int(fid),
            "count": int(count),
        }
        for fid, count in count_dict.items()
    ]

    rows.sort(
        key=lambda row: row["count"],
        reverse=True,
    )

    return rows[:top_n]


def sorted_activation_dict(
    count_dict,
    activation_dict,
    stats,
    top_n=50,
):
    rows = []

    for fid, count in count_dict.items():
        fid = int(fid)

        activation_sum = float(
            activation_dict.get(
                fid,
                0.0,
            )
        )

        rows.append({
            "feature_id":
                fid,
            "count":
                int(count),
            "activation_sum":
                activation_sum,
            "avg_activation_when_selected":
                activation_sum
                / max(int(count), 1),
            "purity":
                float(
                    stats[
                        "label_purity"
                    ][fid].item()
                ),
            "majority_label":
                int(
                    stats[
                        "majority_label"
                    ][fid].item()
                ),
            "valid_count":
                int(
                    stats[
                        "valid_count"
                    ][fid].item()
                ),
            "mean_activation":
                float(
                    stats[
                        "mean_acts"
                    ][fid].item()
                ),
            "sparsity":
                float(
                    stats[
                        "sparsity"
                    ][fid].item()
                ),
        })

    rows.sort(
        key=lambda row: (
            row["count"],
            row[
                "avg_activation_when_selected"
            ],
            row["purity"],
        ),
        reverse=True,
    )

    return rows[:top_n]


def bootstrap_ci(
    values,
    n_bootstrap=10000,
    seed=42,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return None, None

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(
            n_bootstrap,
            len(values),
        ),
        replace=True,
    )

    means = samples.mean(axis=1)

    low, high = np.percentile(
        means,
        [2.5, 97.5],
    )

    return float(low), float(high)


def paired_tests(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return None, None, 0, 0, None

    positive = int(
        (values > 0).sum()
    )

    negative = int(
        (values < 0).sum()
    )

    nonzero = positive + negative

    fraction_positive = (
        positive / len(values)
    )

    if np.allclose(values, 0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(
            wilcoxon(
                values,
                alternative="greater",
                zero_method="wilcox",
            ).pvalue
        )

    if nonzero:
        sign_p = float(
            binomtest(
                positive,
                n=nonzero,
                p=0.5,
                alternative="greater",
            ).pvalue
        )
    else:
        sign_p = 1.0

    return (
        wilcoxon_p,
        sign_p,
        positive,
        negative,
        float(fraction_positive),
    )


def summarize_treated(
    per_image_results,
    bootstrap_samples,
    seed,
):
    treated = [
        row
        for row in per_image_results
        if row[
            "number_target_neurons_ablated"
        ] > 0
    ]

    if not treated:
        return {
            "total_treated_images": 0,
            "fraction_treated": 0.0,
            "mean_targeted_confidence_drop": None,
            "mean_random_confidence_drop": None,
            "targeted_minus_random_drop": None,
            "mean_targeted_minus_random_logit_drop": None,
            "median_targeted_minus_random_logit_drop": None,
            "fraction_targeted_drop_greater_than_random": None,
            "mean_targeted_minus_random_margin_drop": None,
            "median_targeted_minus_random_margin_drop": None,
            "fraction_targeted_margin_drop_greater_than_random": None,
            "ci_95_low": None,
            "ci_95_high": None,
            "wilcoxon_p": None,
            "sign_test_p": None,
            "positive_images": 0,
            "negative_images": 0,
        }

    conf_differences = np.array([
        row["targeted_drop"]
        - row["random_drop"]
        for row in treated
    ])

    logit_differences = np.array([
        row["logit_drop_difference"]
        for row in treated
    ])

    margin_differences = np.array([
        row["margin_drop_difference"]
        for row in treated
    ])

    ci_low, ci_high = bootstrap_ci(
        margin_differences,
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )

    (
        wilcoxon_p,
        sign_p,
        positive,
        negative,
        fraction_positive,
    ) = paired_tests(
        margin_differences
    )

    return {
        "total_treated_images":
            len(treated),

        "mean_targeted_confidence_drop":
            float(
                np.mean([
                    row["targeted_drop"]
                    for row in treated
                ])
            ),

        "mean_random_confidence_drop":
            float(
                np.mean([
                    row["random_drop"]
                    for row in treated
                ])
            ),

        "targeted_minus_random_drop":
            float(
                conf_differences.mean()
            ),

        "mean_targeted_minus_random_logit_drop":
            float(
                logit_differences.mean()
            ),

        "median_targeted_minus_random_logit_drop":
            float(
                np.median(
                    logit_differences
                )
            ),

        "fraction_targeted_drop_greater_than_random":
            float(
                (
                    logit_differences > 0
                ).mean()
            ),

        "mean_targeted_minus_random_margin_drop":
            float(
                margin_differences.mean()
            ),

        "median_targeted_minus_random_margin_drop":
            float(
                np.median(
                    margin_differences
                )
            ),

        "fraction_targeted_margin_drop_greater_than_random":
            fraction_positive,

        "ci_95_low":
            ci_low,

        "ci_95_high":
            ci_high,

        "wilcoxon_p":
            wilcoxon_p,

        "sign_test_p":
            sign_p,

        "positive_images":
            positive,

        "negative_images":
            negative,
    }


def run_class(
    target_class,
    target_class_name,
    target_indices,
    target_feature_ids,
    test_dataset,
    model,
    ksae,
    stats,
    cfg,
    device,
):
    loader = DataLoader(
        Subset(
            test_dataset,
            target_indices,
        ),
        batch_size=cfg["batch_size"],
        shuffle=False,
        pin_memory=False,
    )

    class_cfg = {
        "target_class":
            target_class,

        "target_class_name":
            target_class_name,

        "target_feature_ids":
            target_feature_ids,

        "min_purity":
            cfg["min_purity"],

        "min_valid":
            cfg["min_valid"],

        "max_ablate_per_image":
            cfg["max_ablate_per_image"],

        "active_feature_ranking":
            cfg["active_feature_ranking"],
    }

    metrics = {
        "original":
            empty_metrics(),

        "sae_reconstruction":
            empty_metrics(),

        "active_targeted_ablation":
            empty_metrics(),

        "active_random_ablation":
            empty_metrics(),
    }

    feature_counter = {
        "total_active_features": 0,
        "total_target_ablated": 0,
        "total_random_ablated": 0,
        "images_with_target_features": 0,
        "target_feature_counts": {},
        "target_feature_activation_sums": {},
        "random_feature_counts": {},
    }

    per_image_results = []
    sample_offset = 0

    # Makes tie-breaking reproducible independently for each class.
    torch.manual_seed(
        cfg["random_seed"]
        + target_class
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            cfg["random_seed"]
            + target_class
        )

    with torch.no_grad():
        for batch_idx, (
            diffusion_images,
            clip_images,
            labels,
            _,
        ) in enumerate(loader):

            diffusion_images = (
                diffusion_images.to(device)
            )

            labels = labels.to(device)

            features = model.get_features(
                diffusion_images,
                None,
                cfg["diffusion_timestep"],
            )

            logits_original = (
                model.classifer(features)
            )

            (
                features_recon,
                features_target,
                features_random,
                interventions,
            ) = build_feature_variants(
                features,
                ksae,
                stats,
                class_cfg,
                feature_counter,
            )

            logits_recon = model.classifer(
                features_recon
            )

            logits_target = model.classifer(
                features_target
            )

            logits_random = model.classifer(
                features_random
            )

            update_metrics(
                metrics["original"],
                logits_original,
                labels,
                target_class,
            )

            update_metrics(
                metrics[
                    "sae_reconstruction"
                ],
                logits_recon,
                labels,
                target_class,
            )

            update_metrics(
                metrics[
                    "active_targeted_ablation"
                ],
                logits_target,
                labels,
                target_class,
            )

            update_metrics(
                metrics[
                    "active_random_ablation"
                ],
                logits_random,
                labels,
                target_class,
            )

            probs_original = torch.softmax(
                logits_original,
                dim=1,
            )

            probs_target = torch.softmax(
                logits_target,
                dim=1,
            )

            probs_random = torch.softmax(
                logits_random,
                dim=1,
            )

            batch_size_actual = (
                labels.size(0)
            )

            batch_dataset_indices = (
                target_indices[
                    sample_offset:
                    sample_offset
                    + batch_size_actual
                ]
            )

            for i in range(
                batch_size_actual
            ):
                label = int(
                    labels[i].item()
                )

                competitor_mask = torch.ones(
                    logits_original.size(1),
                    dtype=torch.bool,
                    device=device,
                )

                competitor_mask[label] = False

                original_conf = float(
                    probs_original[
                        i, label
                    ].item()
                )

                targeted_conf = float(
                    probs_target[
                        i, label
                    ].item()
                )

                random_conf = float(
                    probs_random[
                        i, label
                    ].item()
                )

                original_logit = float(
                    logits_original[
                        i, label
                    ].item()
                )

                targeted_logit = float(
                    logits_target[
                        i, label
                    ].item()
                )

                random_logit = float(
                    logits_random[
                        i, label
                    ].item()
                )

                original_competing_logit = float(
                    logits_original[
                        i,
                        competitor_mask,
                    ].max().item()
                )

                targeted_competing_logit = float(
                    logits_target[
                        i,
                        competitor_mask,
                    ].max().item()
                )

                random_competing_logit = float(
                    logits_random[
                        i,
                        competitor_mask,
                    ].max().item()
                )

                original_margin = (
                    original_logit
                    - original_competing_logit
                )

                targeted_margin = (
                    targeted_logit
                    - targeted_competing_logit
                )

                random_margin = (
                    random_logit
                    - random_competing_logit
                )

                targeted_drop = (
                    original_conf
                    - targeted_conf
                )

                random_drop = (
                    original_conf
                    - random_conf
                )

                targeted_logit_drop = (
                    original_logit
                    - targeted_logit
                )

                random_logit_drop = (
                    original_logit
                    - random_logit
                )

                targeted_margin_drop = (
                    original_margin
                    - targeted_margin
                )

                random_margin_drop = (
                    original_margin
                    - random_margin
                )

                intervention = (
                    interventions[i]
                )

                per_image_results.append({
                    "image_index":
                        int(
                            batch_dataset_indices[
                                i
                            ]
                        ),

                    "true_label":
                        label,

                    "original_pred":
                        int(
                            logits_original[
                                i
                            ].argmax().item()
                        ),

                    "targeted_pred":
                        int(
                            logits_target[
                                i
                            ].argmax().item()
                        ),

                    "random_pred":
                        int(
                            logits_random[
                                i
                            ].argmax().item()
                        ),

                    "original_confidence":
                        original_conf,

                    "targeted_confidence":
                        targeted_conf,

                    "random_confidence":
                        random_conf,

                    "targeted_drop":
                        targeted_drop,

                    "random_drop":
                        random_drop,

                    "original_true_logit":
                        original_logit,

                    "targeted_true_logit":
                        targeted_logit,

                    "random_true_logit":
                        random_logit,

                    "targeted_logit_drop":
                        targeted_logit_drop,

                    "random_logit_drop":
                        random_logit_drop,

                    "logit_drop_difference":
                        targeted_logit_drop
                        - random_logit_drop,

                    "original_margin":
                        original_margin,

                    "targeted_margin":
                        targeted_margin,

                    "random_margin":
                        random_margin,

                    "targeted_margin_drop":
                        targeted_margin_drop,

                    "random_margin_drop":
                        random_margin_drop,

                    "margin_drop_difference":
                        targeted_margin_drop
                        - random_margin_drop,

                    **intervention,
                })

            sample_offset += (
                batch_size_actual
            )

            counts = [
                x[
                    "number_target_neurons_ablated"
                ]
                for x in interventions
            ]

            print(
                f"  Batch {batch_idx}: "
                f"target_counts={counts}"
            )

    finalized_metrics = {
        name: finalize_metrics(value)
        for name, value
        in metrics.items()
    }

    total_images = (
        finalized_metrics[
            "original"
        ]["total"]
    )

    original_conf = (
        finalized_metrics[
            "original"
        ][
            "true_class_confidence"
        ]
    )

    treated_summary = summarize_treated(
        per_image_results,
        bootstrap_samples=cfg.get(
            "bootstrap_samples",
            10000,
        ),
        seed=(
            cfg["random_seed"]
            + target_class
        ),
    )

    treated_summary["fraction_treated"] = (
        treated_summary[
            "total_treated_images"
        ]
        / total_images
        if total_images
        else 0.0
    )

    results = {
        "class_id":
            target_class,

        "class_name":
            target_class_name,

        "selected_feature_count":
            len(target_feature_ids),

        "selected_feature_ids":
            target_feature_ids,

        "metrics":
            finalized_metrics,

        "confidence_drop_from_original": {
            "sae_reconstruction":
                original_conf
                - finalized_metrics[
                    "sae_reconstruction"
                ][
                    "true_class_confidence"
                ],

            "active_targeted_ablation":
                original_conf
                - finalized_metrics[
                    "active_targeted_ablation"
                ][
                    "true_class_confidence"
                ],

            "active_random_ablation":
                original_conf
                - finalized_metrics[
                    "active_random_ablation"
                ][
                    "true_class_confidence"
                ],
        },

        "activation_summary": {
            "total_images":
                total_images,

            "ksae_k":
                ksae["k"],

            "total_active_features_seen":
                feature_counter[
                    "total_active_features"
                ],

            "total_target_ablated":
                feature_counter[
                    "total_target_ablated"
                ],

            "total_random_ablated":
                feature_counter[
                    "total_random_ablated"
                ],

            "images_with_target_features":
                feature_counter[
                    "images_with_target_features"
                ],

            "fraction_images_with_target_features":
                (
                    feature_counter[
                        "images_with_target_features"
                    ]
                    / total_images
                    if total_images
                    else 0.0
                ),

            "avg_target_ablated_per_image":
                (
                    feature_counter[
                        "total_target_ablated"
                    ]
                    / total_images
                    if total_images
                    else 0.0
                ),

            "avg_random_ablated_per_image":
                (
                    feature_counter[
                        "total_random_ablated"
                    ]
                    / total_images
                    if total_images
                    else 0.0
                ),

            "random_control":
                (
                    "active non-target-class features "
                    "matched without replacement on "
                    "absolute activation magnitude"
                ),
        },

        "treated_images_summary":
            treated_summary,

        "top_target_ablated_features":
            sorted_activation_dict(
                feature_counter[
                    "target_feature_counts"
                ],
                feature_counter[
                    "target_feature_activation_sums"
                ],
                stats,
            ),

        "top_random_ablated_features":
            sorted_count_dict(
                feature_counter[
                    "random_feature_counts"
                ]
            ),

        "per_image_results":
            per_image_results,
    }

    return results


def make_summary_row(result):
    treated = (
        result["treated_images_summary"]
    )

    metrics = result["metrics"]

    return {
        "class_id":
            result["class_id"],

        "class_name":
            result["class_name"],

        "n_selected_features":
            result["selected_feature_count"],

        "n_test_images":
            metrics["original"]["total"],

        "n_treated":
            treated[
                "total_treated_images"
            ],

        "fraction_treated":
            treated["fraction_treated"],

        "mean_confidence_difference":
            treated[
                "targeted_minus_random_drop"
            ],

        "mean_logit_difference":
            treated[
                "mean_targeted_minus_random_logit_drop"
            ],

        "median_logit_difference":
            treated[
                "median_targeted_minus_random_logit_drop"
            ],

        "mean_margin_difference":
            treated[
                "mean_targeted_minus_random_margin_drop"
            ],

        "median_margin_difference":
            treated[
                "median_targeted_minus_random_margin_drop"
            ],

        "ci_95_low":
            treated["ci_95_low"],

        "ci_95_high":
            treated["ci_95_high"],

        "positive_images":
            treated["positive_images"],

        "negative_images":
            treated["negative_images"],

        "fraction_positive":
            treated[
                "fraction_targeted_margin_drop_greater_than_random"
            ],

        "wilcoxon_p":
            treated["wilcoxon_p"],

        "sign_test_p":
            treated["sign_test_p"],

        "original_accuracy":
            metrics["original"][
                "accuracy"
            ],

        "targeted_accuracy":
            metrics[
                "active_targeted_ablation"
            ]["accuracy"],

        "random_accuracy":
            metrics[
                "active_random_ablation"
            ]["accuracy"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    cfg = load_json(args.config)

    sys.path.insert(
        0,
        cfg["diffc_dir"],
    )

    sys.path.insert(
        1,
        cfg["sd_ksae_dir"],
    )

    from helpers.dataset import (
        HuggingFaceImageDataset,
        load_huggingface_dataset,
    )

    from constants import (
        model_base_dict,
        diffusion_transformers_val,
        clip_transforms,
    )

    from models import ImageClassifer

    cfg["random_seed"] = cfg.get(
        "random_seed",
        42,
    )

    random.seed(
        cfg["random_seed"]
    )

    np.random.seed(
        cfg["random_seed"]
    )

    torch.manual_seed(
        cfg["random_seed"]
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            cfg["random_seed"]
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    print(
        "Loading feature statistics..."
    )

    stats = load_feature_stats(
        cfg["feature_dir"],
        device,
    )

    feature_summary_df = pd.read_csv(
        cfg["feature_summary_path"]
    )

    print("Loading k-SAE...")

    ksae = load_ksae(
        cfg["ksae_checkpoint_path"],
        device,
        default_k=cfg.get(
            "ksae_k",
            32,
        ),
    )

    print(
        "k-SAE k:",
        ksae["k"],
    )

    print(
        "k-SAE n_features:",
        ksae["n_features"],
    )

    print("Loading dataset...")

    hf_test_dataset = (
        load_huggingface_dataset(
            cfg["dataset_flag"],
            split=cfg.get(
                "split",
                "test",
            ),
        )
    )

    class_names = (
        hf_test_dataset
        .features["label"]
        .names
    )

    labels_all = (
        hf_test_dataset["label"]
    )

    test_dataset = (
        HuggingFaceImageDataset(
            hf_test_dataset,
            diffusion_transformers_val,
            clip_transforms,
        )
    )

    print("Loading DiffC model...")

    diffc_config = {
        "dataset_flag":
            cfg["dataset_flag"],

        "output_dir":
            "",

        "seed":
            cfg["random_seed"],

        "model_name":
            cfg["model_name"],

        "diffusion_timestep":
            cfg["diffusion_timestep"],

        "diffusion_layer":
            cfg["diffusion_layer"],

        "learning_rate":
            cfg.get(
                "learning_rate",
                1e-4,
            ),

        "num_epochs":
            cfg.get(
                "num_epochs",
                90,
            ),

        "batch_size":
            cfg["batch_size"],

        "prompt_type":
            cfg.get(
                "prompt_type",
                "empty",
            ),

        "pooling_strategy":
            cfg.get(
                "pooling_strategy",
                "GAP",
            ),

        "dropout_rate":
            cfg["dropout_rate"],

        "num_classes":
            cfg["num_classes"],

        "num_devices":
            1,

        "feature_model":
            model_base_dict[
                cfg["model_name"]
            ],

        "diffusion_step_type":
            cfg.get(
                "diffusion_step_type",
                "onestep",
            ),

        "device":
            device,

        "input_channels":
            cfg["input_channels"],
    }

    model = ImageClassifer(
        diffc_config
    ).to(device)

    checkpoint = torch.load(
        cfg["diffc_checkpoint_path"],
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Loaded DiffC checkpoint epoch:",
        checkpoint.get("epoch"),
    )

    target_classes_cfg = cfg.get(
        "target_classes",
        "all",
    )

    if target_classes_cfg == "all":
        target_classes = list(
            range(len(class_names))
        )
    else:
        target_classes = [
            int(x)
            for x in target_classes_cfg
        ]

    for class_id in target_classes:
        if not (
            0
            <= class_id
            < len(class_names)
        ):
            raise ValueError(
                f"Invalid target class "
                f"{class_id}. Valid range: "
                f"0..{len(class_names) - 1}"
            )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_root = (
        Path(cfg["output_dir"])
        / f"run_{timestamp}"
    )

    details_dir = (
        output_root / "per_class"
    )

    details_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []
    summary_rows = []

    for class_id in target_classes:
        class_name = (
            class_names[class_id]
        )

        target_feature_ids = (
            select_target_features(
                feature_summary_df,
                class_id,
                cfg["min_purity"],
                cfg["min_valid"],
            )
        )

        target_indices = [
            i
            for i, label
            in enumerate(labels_all)
            if int(label) == class_id
        ][:cfg["max_images"]]

        print()
        print("=" * 70)

        print(
            f"Class {class_id}: "
            f"{class_name}"
        )

        print(
            f"Selected features: "
            f"{len(target_feature_ids)} | "
            f"Test images: "
            f"{len(target_indices)}"
        )

        if (
            not target_feature_ids
            or not target_indices
        ):
            print(
                "Skipping class: no selected "
                "features or no test images."
            )

            result = {
                "class_id":
                    class_id,

                "class_name":
                    class_name,

                "selected_feature_count":
                    len(
                        target_feature_ids
                    ),

                "selected_feature_ids":
                    target_feature_ids,

                "metrics": {
                    "original": {
                        "accuracy": None,
                        "total":
                            len(
                                target_indices
                            ),
                    },

                    "active_targeted_ablation": {
                        "accuracy": None
                    },

                    "active_random_ablation": {
                        "accuracy": None
                    },
                },

                "treated_images_summary": {
                    "total_treated_images":
                        0,

                    "fraction_treated":
                        0.0,

                    "targeted_minus_random_drop":
                        None,

                    "mean_targeted_minus_random_logit_drop":
                        None,

                    "median_targeted_minus_random_logit_drop":
                        None,

                    "mean_targeted_minus_random_margin_drop":
                        None,

                    "median_targeted_minus_random_margin_drop":
                        None,

                    "ci_95_low":
                        None,

                    "ci_95_high":
                        None,

                    "positive_images":
                        0,

                    "negative_images":
                        0,

                    "fraction_targeted_margin_drop_greater_than_random":
                        None,

                    "wilcoxon_p":
                        None,

                    "sign_test_p":
                        None,
                },

                "per_image_results": [],
            }

        else:
            result = run_class(
                class_id,
                class_name,
                target_indices,
                target_feature_ids,
                test_dataset,
                model,
                ksae,
                stats,
                cfg,
                device,
            )

        detail_path = (
            details_dir
            / (
                f"class_{class_id:02d}_"
                f"{class_name}.json"
            )
        )

        with detail_path.open(
            "w"
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )

        all_results.append({
            "class_id":
                class_id,

            "class_name":
                class_name,

            "detail_file":
                str(detail_path),

            "treated_images_summary":
                result[
                    "treated_images_summary"
                ],
        })

        summary_rows.append(
            make_summary_row(
                result
            )
        )

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values("class_id")
    )

    summary_csv = (
        output_root
        / "causal_statistical_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    combined_json = (
        output_root
        / "combined_summary.json"
    )

    with combined_json.open(
        "w"
    ) as f:
        json.dump({
            "run_timestamp":
                timestamp,

            "config":
                cfg,

            "classes":
                all_results,
        }, f, indent=2)

    print()
    print("=" * 70)

    display_columns = [
        "class_id",
        "class_name",
        "n_selected_features",
        "n_treated",
        "mean_margin_difference",
        "median_margin_difference",
        "ci_95_low",
        "ci_95_high",
        "fraction_positive",
        "wilcoxon_p",
        "sign_test_p",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Saved run to:",
        output_root,
    )

    print(
        "Summary CSV:",
        summary_csv,
    )

    print(
        "Combined JSON:",
        combined_json,
    )


if __name__ == "__main__":
    main()