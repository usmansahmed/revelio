import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

import pandas as pd
import torch
import torch.nn.functional as F
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


def load_target_feature_ids(
    summary_path,
    target_class,
    min_purity,
    min_valid,
):
    df = pd.read_csv(summary_path)

    selected = df[
        (df["majority_label"] == target_class)
        & (df["label_purity"] >= min_purity)
        & (df["valid_top_count"] >= min_valid)
    ]

    feature_ids = selected["feature_id"].astype(int).tolist()

    print(
        f"Automatically selected {len(feature_ids)} "
        f"features for class {target_class}"
    )
    print("Feature IDs:", feature_ids)

    return feature_ids


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
    # k-SAE uses TopK directly. Do not apply ReLU before TopK.
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

    # A target neuron must:
    # 1. currently be active in this image
    # 2. belong to our selected class-specific feature set
    # 3. satisfy the purity and valid-count thresholds
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

    # Random control neurons must also be active, but must not
    # be associated with the target class.
    control_pool = active_ids[
        stats["majority_label"][active_ids]
        != target_class
    ]

    matched_random_ids = []
    available = control_pool

    # Match controls to targeted neurons based on activation
    # magnitude, without replacement.
    for target_id in target_ids:
        if available.numel() == 0:
            break

        distances = (
            sparse_row[available].abs()
            - sparse_row[target_id].abs()
        ).abs()

        # Tiny random jitter prevents systematic preference for
        # lower feature IDs in exact ties.
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
    cfg,
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
            target_class=cfg["target_class"],
            min_purity=cfg["min_purity"],
            min_valid=cfg["min_valid"],
            max_ablate_per_image=(
                cfg["max_ablate_per_image"]
            ),
            ranking_method=(
                cfg["active_feature_ranking"]
            ),
            target_feature_ids=(
                cfg["target_feature_ids"]
            ),
        )

        target_activation_sum = 0.0

        if target_ids.numel() > 0:
            target_activation_sum = float(
                sparse_acts[
                    i, target_ids
                ].sum().detach().cpu().item()
            )

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
                    ].detach().cpu().item()
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

    # SAE reconstruction variant.
    recon_features = (
        features
        + (recon_pooled - pooled)[
            :, :, None, None
        ]
    )

    # Residual-preserving targeted intervention:
    # change only the decoder directions caused by ablating
    # the selected neurons.
    target_features = (
        features
        + (target_pooled - recon_pooled)[
            :, :, None, None
        ]
    )

    # Same intervention principle for matched random neurons.
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
        fid_int = int(fid)

        activation_sum = float(
            activation_dict.get(
                fid_int,
                0.0,
            )
        )

        rows.append({
            "feature_id": fid_int,
            "count": int(count),
            "activation_sum": activation_sum,
            "avg_activation_when_selected":
                activation_sum
                / max(int(count), 1),
            "purity": float(
                stats["label_purity"][
                    fid_int
                ].detach().cpu().item()
            ),
            "majority_label": int(
                stats["majority_label"][
                    fid_int
                ].detach().cpu().item()
            ),
            "valid_count": int(
                stats["valid_count"][
                    fid_int
                ].detach().cpu().item()
            ),
            "mean_activation": float(
                stats["mean_acts"][
                    fid_int
                ].detach().cpu().item()
            ),
            "sparsity": float(
                stats["sparsity"][
                    fid_int
                ].detach().cpu().item()
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    args = parser.parse_args()

    cfg = load_json(args.config)

    # Automatically obtain target-class neurons from the
    # feature-summary CSV.
    cfg["target_feature_ids"] = (
        load_target_feature_ids(
            cfg["feature_summary_path"],
            cfg["target_class"],
            cfg["min_purity"],
            cfg["min_valid"],
        )
    )

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

    random_seed = cfg.get(
        "random_seed",
        42,
    )

    random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            random_seed
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    print()
    print("Loading feature statistics...")

    stats = load_feature_stats(
        cfg["feature_dir"],
        device=device,
    )

    print("Loading k-SAE...")

    ksae = load_ksae(
        cfg["ksae_checkpoint_path"],
        device=device,
        default_k=cfg.get(
            "ksae_k",
            32,
        ),
    )

    print("k-SAE k:", ksae["k"])
    print(
        "k-SAE n_features:",
        ksae["n_features"],
    )

    print()
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

    target_class = cfg["target_class"]

    if not 0 <= target_class < len(
        class_names
    ):
        raise ValueError(
            f"target_class {target_class} "
            f"is outside dataset label range "
            f"0..{len(class_names) - 1}."
        )

    target_class_name = (
        class_names[target_class]
    )

    print(
        "Target class:",
        target_class,
        target_class_name,
    )

    labels_all = (
        hf_test_dataset["label"]
    )

    target_indices = [
        i
        for i, label in enumerate(labels_all)
        if int(label) == target_class
    ][:cfg["max_images"]]

    if len(target_indices) == 0:
        raise RuntimeError(
            f"No images found for target "
            f"class {target_class}."
        )

    print(
        "Selected target-class images:",
        len(target_indices),
    )

    print(
        "First indices:",
        target_indices[:20],
    )

    test_dataset = HuggingFaceImageDataset(
        hf_test_dataset,
        diffusion_transformers_val,
        clip_transforms,
    )

    loader = DataLoader(
        Subset(
            test_dataset,
            target_indices,
        ),
        batch_size=cfg["batch_size"],
        shuffle=False,
        pin_memory=False,
    )

    print()
    print("Loading DiffC model...")

    diffc_config = {
        "dataset_flag":
            cfg["dataset_flag"],
        "output_dir":
            "",
        "seed":
            random_seed,
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

    print(
        "Loaded DiffC checkpoint test accuracy:",
        checkpoint.get(
            "test_accuracy"
        ),
    )

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
                cfg[
                    "diffusion_timestep"
                ],
            )

            logits_original = (
                model.classifer(features)
            )

            (
                features_recon,
                features_target,
                features_random,
                batch_interventions,
            ) = build_feature_variants(
                features,
                ksae,
                stats,
                cfg,
                feature_counter,
            )

            logits_recon = (
                model.classifer(
                    features_recon
                )
            )

            logits_target = (
                model.classifer(
                    features_target
                )
            )

            logits_random = (
                model.classifer(
                    features_random
                )
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

            # Per-image results
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

                original_logit = float(logits_original[i, label].item())
                targeted_logit = float(logits_target[i, label].item())
                random_logit = float(logits_random[i, label].item())

                targeted_logit_drop = original_logit - targeted_logit
                random_logit_drop = original_logit - random_logit

                intervention = (
                    batch_interventions[i]
                )

                per_image_results.append({
                    "image_index": int(
                        batch_dataset_indices[
                            i
                        ]
                    ),
                    "true_label":
                        label,
                    "original_confidence":
                        original_conf,
                    "targeted_confidence":
                        targeted_conf,
                    "random_confidence":
                        random_conf,
                    "targeted_drop":
                        original_conf
                        - targeted_conf,
                    "random_drop":
                        original_conf
                        - random_conf,
                    "number_target_neurons_ablated":
                        intervention[
                            "number_target_neurons_ablated"
                        ],
                    "number_random_neurons_ablated":
                        intervention[
                            "number_random_neurons_ablated"
                        ],
                    "target_neuron_ids":
                        intervention[
                            "target_neuron_ids"
                        ],
                    "random_neuron_ids":
                        intervention[
                            "random_neuron_ids"
                        ],
                    "sum_target_activation":
                        intervention[
                            "sum_target_activation"
                        ],
                    "original_true_logit": original_logit,
                    "targeted_true_logit": targeted_logit,
                    "random_true_logit": random_logit,
                    "targeted_logit_drop": targeted_logit_drop,
                    "random_logit_drop": random_logit_drop,
                    "logit_drop_difference": targeted_logit_drop - random_logit_drop,
                })

            sample_offset += (
                batch_size_actual
            )

            target_counts = [x["number_target_neurons_ablated"] for x in batch_interventions]
            print(f"Batch {batch_idx}: " f"target_counts={target_counts}")

    finalized_metrics = {
        name: finalize_metrics(
            value
        )
        for name, value
        in metrics.items()
    }

    original_conf = (
        finalized_metrics[
            "original"
        ][
            "true_class_confidence"
        ]
    )

    recon_conf = (
        finalized_metrics[
            "sae_reconstruction"
        ][
            "true_class_confidence"
        ]
    )

    target_conf = (
        finalized_metrics[
            "active_targeted_ablation"
        ][
            "true_class_confidence"
        ]
    )

    random_conf = (
        finalized_metrics[
            "active_random_ablation"
        ][
            "true_class_confidence"
        ]
    )

    total_images = (
        finalized_metrics[
            "original"
        ]["total"]
    )

    # Only images where a targeted intervention actually occurred.
    treated_images = [
        row
        for row in per_image_results
        if row[
            "number_target_neurons_ablated"
        ] > 0
    ]

    if treated_images:
        logit_differences = [
            row["targeted_logit_drop"] - row["random_logit_drop"]
            for row in treated_images
        ]

        mean_logit_difference = sum(logit_differences) / len(logit_differences)
        median_logit_difference = median(logit_differences)

        fraction_targeted_greater = (
            sum(
                row["targeted_logit_drop"] > row["random_logit_drop"]
                for row in treated_images
            )
            / len(treated_images)
        )
    else:
        mean_logit_difference = 0.0
        median_logit_difference = 0.0
        fraction_targeted_greater = 0.0

    if treated_images:
        treated_target_drop = (
            sum(
                row["targeted_drop"]
                for row
                in treated_images
            )
            / len(treated_images)
        )

        treated_random_drop = (
            sum(
                row["random_drop"]
                for row
                in treated_images
            )
            / len(treated_images)
        )

    else:
        treated_target_drop = 0.0
        treated_random_drop = 0.0

    results = {
        "config": {
            **cfg,
            "target_class_name":
                target_class_name,
        },

        "metrics":
            finalized_metrics,

        "confidence_drop_from_original": {
            "sae_reconstruction":
                original_conf
                - recon_conf,
            "active_targeted_ablation":
                original_conf
                - target_conf,
            "active_random_ablation":
                original_conf
                - random_conf,
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
                feature_counter[
                    "images_with_target_features"
                ]
                / total_images,
            "avg_target_ablated_per_image":
                feature_counter[
                    "total_target_ablated"
                ]
                / total_images,
            "avg_random_ablated_per_image":
                feature_counter[
                    "total_random_ablated"
                ]
                / total_images,
            "random_control":
                (
                    "active non-target-class "
                    "features matched without "
                    "replacement on absolute "
                    "activation magnitude"
                ),
        },

        "treated_images_summary": {
            "total_treated_images":
                len(treated_images),

            "fraction_treated":
                len(treated_images)
                / total_images,

            "mean_targeted_confidence_drop":
                treated_target_drop,

            "mean_random_confidence_drop":
                treated_random_drop,

            "targeted_minus_random_drop":
                treated_target_drop
                - treated_random_drop,
            "mean_targeted_minus_random_logit_drop": mean_logit_difference,
            "median_targeted_minus_random_logit_drop": median_logit_difference,
            "fraction_targeted_drop_greater_than_random": fraction_targeted_greater,
        },

        "top_target_ablated_features":
            sorted_activation_dict(
                feature_counter[
                    "target_feature_counts"
                ],
                feature_counter[
                    "target_feature_activation_sums"
                ],
                stats,
                top_n=50,
            ),

        "top_random_ablated_features":
            sorted_count_dict(
                feature_counter[
                    "random_feature_counts"
                ],
                top_n=50,
            ),

        "per_image_results":
            per_image_results,
    }

    # Save every experiment separately.
    base_output_path = Path(
        cfg["output_path"]
    )

    base_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        base_output_path.with_name(
            f"{base_output_path.stem}_"
            f"{timestamp}"
            f"{base_output_path.suffix}"
        )
    )

    with output_path.open("w") as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    print()
    print(json.dumps(
        results,
        indent=4,
    ))

    print()
    print(
        "Saved results to:",
        output_path,
    )


if __name__ == "__main__":
    main()