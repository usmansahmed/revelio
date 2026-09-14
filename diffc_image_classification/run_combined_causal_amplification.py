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
        "label_purity": load_tensor(feature_dir / "label_purity_top10.pt", device),
        "majority_label": load_tensor(feature_dir / "majority_label_top10.pt", device),
        "valid_count": load_tensor(feature_dir / "valid_top_count_top10.pt", device),
        "mean_acts": load_tensor(feature_dir / "sae_mean_acts.pt", device),
        "sparsity": load_tensor(feature_dir / "sae_sparsity.pt", device),
    }


def load_ksae(checkpoint_path, device, default_k=32):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["state_dict"]
    k = checkpoint["cfg"].k if "cfg" in checkpoint and hasattr(checkpoint["cfg"], "k") else default_k
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
    pre_acts = (x - ksae["b_dec"]) @ ksae["W_enc"] + ksae["b_enc"]
    top_values, top_indices = torch.topk(pre_acts, k=ksae["k"], dim=-1)
    sparse_acts = torch.zeros_like(pre_acts)
    sparse_acts.scatter_(dim=-1, index=top_indices, src=top_values)
    return sparse_acts


def ksae_decode(sparse_acts, ksae):
    return sparse_acts @ ksae["W_dec"] + ksae["b_dec"]


def select_target_features(summary_df, target_class, min_purity, min_valid):
    selected = summary_df[
        (summary_df["majority_label"] == target_class)
        & (summary_df["label_purity"] >= min_purity)
        & (summary_df["valid_top_count"] >= min_valid)
    ]
    return selected["feature_id"].astype(int).tolist()


def score_active_candidates(active_ids, sparse_row, stats, ranking_method):
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
    raise ValueError(f"Unknown active_feature_ranking: {ranking_method}")


def select_active_class_features(
    sparse_row, stats, target_class, min_purity, min_valid,
    max_amplify_per_image, ranking_method, target_feature_ids,
):
    active_ids = torch.nonzero(sparse_row != 0, as_tuple=False).flatten()
    if active_ids.numel() == 0:
        return active_ids, active_ids, active_ids

    allowed_ids = torch.tensor(target_feature_ids, device=active_ids.device, dtype=active_ids.dtype)
    class_mask = (
        torch.isin(active_ids, allowed_ids)
        & (stats["majority_label"][active_ids] == target_class)
        & (stats["label_purity"][active_ids] >= min_purity)
        & (stats["valid_count"][active_ids] >= min_valid)
    )
    target_ids = active_ids[class_mask]

    if target_ids.numel() > max_amplify_per_image:
        scores = score_active_candidates(target_ids, sparse_row, stats, ranking_method)
        order = torch.argsort(scores, descending=True)
        target_ids = target_ids[order[:max_amplify_per_image]]

    # Match each target neuron to an active non-target neuron with similar |activation|.
    control_pool = active_ids[stats["majority_label"][active_ids] != target_class]
    matched_random_ids = []
    available = control_pool

    for target_id in target_ids:
        if available.numel() == 0:
            break
        distances = (sparse_row[available].abs() - sparse_row[target_id].abs()).abs()
        distances = distances + torch.rand_like(distances) * 1e-12  # tie-breaking only
        match_position = torch.argmin(distances)
        matched_random_ids.append(available[match_position])
        available = torch.cat((available[:match_position], available[match_position + 1:]))

    random_ids = torch.stack(matched_random_ids) if matched_random_ids else control_pool[:0]
    return active_ids, target_ids, random_ids


def build_feature_variants(features, ksae, stats, class_cfg, feature_counter):
    pooled = features.mean(dim=(2, 3))
    sparse_acts = ksae_encode(pooled, ksae)
    sparse_target = sparse_acts.clone()
    sparse_random = sparse_acts.clone()
    factor = class_cfg["amplification_factor"]
    batch_interventions = []

    for i in range(sparse_acts.size(0)):
        active_ids, target_ids, random_ids = select_active_class_features(
            sparse_acts[i], stats,
            target_class=class_cfg["target_class"],
            min_purity=class_cfg["min_purity"],
            min_valid=class_cfg["min_valid"],
            max_amplify_per_image=class_cfg["max_amplify_per_image"],
            ranking_method=class_cfg["active_feature_ranking"],
            target_feature_ids=class_cfg["target_feature_ids"],
        )

        target_original = sparse_acts[i, target_ids].clone() if target_ids.numel() else sparse_acts.new_empty(0)
        random_original = sparse_acts[i, random_ids].clone() if random_ids.numel() else sparse_acts.new_empty(0)

        if target_ids.numel() > 0:
            sparse_target[i, target_ids] = target_original * factor
        if random_ids.numel() > 0:
            sparse_random[i, random_ids] = random_original * factor

        target_amplified = sparse_target[i, target_ids] if target_ids.numel() else sparse_acts.new_empty(0)
        random_amplified = sparse_random[i, random_ids] if random_ids.numel() else sparse_acts.new_empty(0)

        batch_interventions.append({
            "target_neuron_ids": [int(x) for x in target_ids.detach().cpu().tolist()],
            "random_neuron_ids": [int(x) for x in random_ids.detach().cpu().tolist()],
            "number_target_neurons_amplified": int(target_ids.numel()),
            "number_random_neurons_amplified": int(random_ids.numel()),
            "target_original_activations": [float(x) for x in target_original.detach().cpu().tolist()],
            "target_amplified_activations": [float(x) for x in target_amplified.detach().cpu().tolist()],
            "random_original_activations": [float(x) for x in random_original.detach().cpu().tolist()],
            "random_amplified_activations": [float(x) for x in random_amplified.detach().cpu().tolist()],
            "sum_target_original_activation": float(target_original.sum().item()) if target_original.numel() else 0.0,
            "sum_random_original_activation": float(random_original.sum().item()) if random_original.numel() else 0.0,
        })

        for fid in target_ids.detach().cpu().tolist():
            fid = int(fid)
            feature_counter["target_feature_counts"][fid] = feature_counter["target_feature_counts"].get(fid, 0) + 1
            feature_counter["target_feature_activation_sums"][fid] = (
                feature_counter["target_feature_activation_sums"].get(fid, 0.0) + float(sparse_acts[i, fid].item())
            )

        for fid in random_ids.detach().cpu().tolist():
            fid = int(fid)
            feature_counter["random_feature_counts"][fid] = feature_counter["random_feature_counts"].get(fid, 0) + 1

        feature_counter["total_active_features"] += int(active_ids.numel())
        feature_counter["total_target_amplified"] += int(target_ids.numel())
        feature_counter["total_random_amplified"] += int(random_ids.numel())
        if target_ids.numel() > 0:
            feature_counter["images_with_target_features"] += 1

    recon_pooled = ksae_decode(sparse_acts, ksae)
    target_pooled = ksae_decode(sparse_target, ksae)
    random_pooled = ksae_decode(sparse_random, ksae)

    # Diagnostic SAE reconstruction baseline.
    recon_features = features + (recon_pooled - pooled)[:, :, None, None]

    # Residual-preserving amplification: inject only the change caused by scaling neurons.
    target_features = features + (target_pooled - recon_pooled)[:, :, None, None]
    random_features = features + (random_pooled - recon_pooled)[:, :, None, None]
    return recon_features, target_features, random_features, batch_interventions


def empty_metrics():
    return {"loss_sum": 0.0, "correct": 0, "total": 0, "true_conf_sum": 0.0, "target_conf_sum": 0.0}


def update_metrics(metrics, logits, labels, target_class):
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    metrics["loss_sum"] += F.cross_entropy(logits, labels, reduction="sum").item()
    metrics["correct"] += (preds == labels).sum().item()
    metrics["total"] += labels.numel()
    metrics["true_conf_sum"] += probs[torch.arange(labels.numel(), device=labels.device), labels].sum().item()
    metrics["target_conf_sum"] += probs[:, target_class].sum().item()


def finalize_metrics(metrics):
    total = metrics["total"]
    if total == 0:
        return {"accuracy": 0.0, "cross_entropy": 0.0, "true_class_confidence": 0.0, "target_class_confidence": 0.0, "total": 0}
    return {
        "accuracy": metrics["correct"] / total,
        "cross_entropy": metrics["loss_sum"] / total,
        "true_class_confidence": metrics["true_conf_sum"] / total,
        "target_class_confidence": metrics["target_conf_sum"] / total,
        "total": total,
    }


def sorted_count_dict(count_dict, top_n=50):
    rows = [{"feature_id": int(fid), "count": int(count)} for fid, count in count_dict.items()]
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:top_n]


def sorted_activation_dict(count_dict, activation_dict, stats, top_n=50):
    rows = []
    for fid, count in count_dict.items():
        fid = int(fid)
        activation_sum = float(activation_dict.get(fid, 0.0))
        rows.append({
            "feature_id": fid,
            "count": int(count),
            "activation_sum": activation_sum,
            "avg_activation_when_selected": activation_sum / max(int(count), 1),
            "purity": float(stats["label_purity"][fid].item()),
            "majority_label": int(stats["majority_label"][fid].item()),
            "valid_count": int(stats["valid_count"][fid].item()),
            "mean_activation": float(stats["mean_acts"][fid].item()),
            "sparsity": float(stats["sparsity"][fid].item()),
        })
    rows.sort(key=lambda row: (row["count"], row["avg_activation_when_selected"], row["purity"]), reverse=True)
    return rows[:top_n]


def bootstrap_ci(values, n_bootstrap=10000, seed=42):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def paired_tests(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None, 0, 0, None

    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    nonzero = positive + negative
    fraction_positive = positive / len(values)

    wilcoxon_p = 1.0 if np.allclose(values, 0) else float(
        wilcoxon(values, alternative="greater", zero_method="wilcox").pvalue
    )
    sign_p = float(binomtest(positive, n=nonzero, p=0.5, alternative="greater").pvalue) if nonzero else 1.0
    return wilcoxon_p, sign_p, positive, negative, float(fraction_positive)


def summarize_amplified(per_image_results, bootstrap_samples, seed):
    amplified = [row for row in per_image_results if row["number_target_neurons_amplified"] > 0]
    if not amplified:
        return {
            "total_amplified_images": 0,
            "fraction_amplified": 0.0,
            "mean_targeted_confidence_increase": None,
            "mean_random_confidence_increase": None,
            "targeted_minus_random_confidence_increase": None,
            "mean_targeted_minus_random_logit_increase": None,
            "median_targeted_minus_random_logit_increase": None,
            "fraction_targeted_logit_increase_greater_than_random": None,
            "mean_targeted_minus_random_margin_increase": None,
            "median_targeted_minus_random_margin_increase": None,
            "fraction_targeted_margin_increase_greater_than_random": None,
            "ci_95_low": None,
            "ci_95_high": None,
            "wilcoxon_p": None,
            "sign_test_p": None,
            "positive_images": 0,
            "negative_images": 0,
        }

    conf_differences = np.array([row["confidence_increase_difference"] for row in amplified])
    logit_differences = np.array([row["logit_increase_difference"] for row in amplified])
    margin_differences = np.array([row["margin_increase_difference"] for row in amplified])
    ci_low, ci_high = bootstrap_ci(margin_differences, n_bootstrap=bootstrap_samples, seed=seed)
    wilcoxon_p, sign_p, positive, negative, fraction_positive = paired_tests(margin_differences)

    return {
        "total_amplified_images": len(amplified),
        "mean_targeted_confidence_increase": float(np.mean([row["targeted_confidence_increase"] for row in amplified])),
        "mean_random_confidence_increase": float(np.mean([row["random_confidence_increase"] for row in amplified])),
        "targeted_minus_random_confidence_increase": float(conf_differences.mean()),
        "mean_targeted_minus_random_logit_increase": float(logit_differences.mean()),
        "median_targeted_minus_random_logit_increase": float(np.median(logit_differences)),
        "fraction_targeted_logit_increase_greater_than_random": float((logit_differences > 0).mean()),
        "mean_targeted_minus_random_margin_increase": float(margin_differences.mean()),
        "median_targeted_minus_random_margin_increase": float(np.median(margin_differences)),
        "fraction_targeted_margin_increase_greater_than_random": fraction_positive,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
        "positive_images": positive,
        "negative_images": negative,
    }


def run_class(target_class, target_class_name, target_indices, target_feature_ids, test_dataset, model, ksae, stats, cfg, device):
    loader = DataLoader(
        Subset(test_dataset, target_indices), batch_size=cfg["batch_size"], shuffle=False, pin_memory=False
    )
    class_cfg = {
        "target_class": target_class,
        "target_class_name": target_class_name,
        "target_feature_ids": target_feature_ids,
        "min_purity": cfg["min_purity"],
        "min_valid": cfg["min_valid"],
        "max_amplify_per_image": cfg.get("max_amplify_per_image", 5),
        "active_feature_ranking": cfg["active_feature_ranking"],
        "amplification_factor": cfg["amplification_factor"],
    }

    metrics = {
        "original": empty_metrics(),
        "sae_reconstruction": empty_metrics(),
        "active_targeted_amplification": empty_metrics(),
        "active_random_amplification": empty_metrics(),
    }
    feature_counter = {
        "total_active_features": 0,
        "total_target_amplified": 0,
        "total_random_amplified": 0,
        "images_with_target_features": 0,
        "target_feature_counts": {},
        "target_feature_activation_sums": {},
        "random_feature_counts": {},
    }

    per_image_results = []
    sample_offset = 0
    torch.manual_seed(cfg["random_seed"] + target_class)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["random_seed"] + target_class)

    with torch.no_grad():
        for batch_idx, (diffusion_images, clip_images, labels, _) in enumerate(loader):
            diffusion_images = diffusion_images.to(device)
            labels = labels.to(device)
            features = model.get_features(diffusion_images, None, cfg["diffusion_timestep"])
            logits_original = model.classifer(features)

            features_recon, features_target, features_random, interventions = build_feature_variants(
                features, ksae, stats, class_cfg, feature_counter
            )
            logits_recon = model.classifer(features_recon)
            logits_target = model.classifer(features_target)
            logits_random = model.classifer(features_random)

            update_metrics(metrics["original"], logits_original, labels, target_class)
            update_metrics(metrics["sae_reconstruction"], logits_recon, labels, target_class)
            update_metrics(metrics["active_targeted_amplification"], logits_target, labels, target_class)
            update_metrics(metrics["active_random_amplification"], logits_random, labels, target_class)

            probs_original = torch.softmax(logits_original, dim=1)
            probs_target = torch.softmax(logits_target, dim=1)
            probs_random = torch.softmax(logits_random, dim=1)
            batch_size_actual = labels.size(0)
            batch_dataset_indices = target_indices[sample_offset:sample_offset + batch_size_actual]

            for i in range(batch_size_actual):
                label = int(labels[i].item())
                competitor_mask = torch.ones(logits_original.size(1), dtype=torch.bool, device=device)
                competitor_mask[label] = False

                original_conf = float(probs_original[i, label].item())
                targeted_conf = float(probs_target[i, label].item())
                random_conf = float(probs_random[i, label].item())
                original_logit = float(logits_original[i, label].item())
                targeted_logit = float(logits_target[i, label].item())
                random_logit = float(logits_random[i, label].item())

                original_competing_logit = float(logits_original[i, competitor_mask].max().item())
                targeted_competing_logit = float(logits_target[i, competitor_mask].max().item())
                random_competing_logit = float(logits_random[i, competitor_mask].max().item())
                original_margin = original_logit - original_competing_logit
                targeted_margin = targeted_logit - targeted_competing_logit
                random_margin = random_logit - random_competing_logit

                targeted_conf_increase = targeted_conf - original_conf
                random_conf_increase = random_conf - original_conf
                targeted_logit_increase = targeted_logit - original_logit
                random_logit_increase = random_logit - original_logit
                targeted_margin_increase = targeted_margin - original_margin
                random_margin_increase = random_margin - original_margin

                per_image_results.append({
                    "image_index": int(batch_dataset_indices[i]),
                    "true_label": label,
                    "original_pred": int(logits_original[i].argmax().item()),
                    "targeted_pred": int(logits_target[i].argmax().item()),
                    "random_pred": int(logits_random[i].argmax().item()),
                    "original_confidence": original_conf,
                    "targeted_confidence": targeted_conf,
                    "random_confidence": random_conf,
                    "targeted_confidence_increase": targeted_conf_increase,
                    "random_confidence_increase": random_conf_increase,
                    "confidence_increase_difference": targeted_conf_increase - random_conf_increase,
                    "original_true_logit": original_logit,
                    "targeted_true_logit": targeted_logit,
                    "random_true_logit": random_logit,
                    "targeted_logit_increase": targeted_logit_increase,
                    "random_logit_increase": random_logit_increase,
                    "logit_increase_difference": targeted_logit_increase - random_logit_increase,
                    "original_margin": original_margin,
                    "targeted_margin": targeted_margin,
                    "random_margin": random_margin,
                    "targeted_margin_increase": targeted_margin_increase,
                    "random_margin_increase": random_margin_increase,
                    "margin_increase_difference": targeted_margin_increase - random_margin_increase,
                    **interventions[i],
                })

            sample_offset += batch_size_actual
            counts = [x["number_target_neurons_amplified"] for x in interventions]
            print(f"  Batch {batch_idx}: target_counts={counts}")

    finalized_metrics = {name: finalize_metrics(value) for name, value in metrics.items()}
    total_images = finalized_metrics["original"]["total"]
    original_conf = finalized_metrics["original"]["true_class_confidence"]
    amplified_summary = summarize_amplified(
        per_image_results,
        bootstrap_samples=cfg.get("bootstrap_samples", 10000),
        seed=cfg["random_seed"] + target_class,
    )
    amplified_summary["fraction_amplified"] = (
        amplified_summary["total_amplified_images"] / total_images if total_images else 0.0
    )

    return {
        "class_id": target_class,
        "class_name": target_class_name,
        "amplification_factor": cfg["amplification_factor"],
        "selected_feature_count": len(target_feature_ids),
        "selected_feature_ids": target_feature_ids,
        "metrics": finalized_metrics,
        "confidence_increase_from_original": {
            "sae_reconstruction": finalized_metrics["sae_reconstruction"]["true_class_confidence"] - original_conf,
            "active_targeted_amplification": finalized_metrics["active_targeted_amplification"]["true_class_confidence"] - original_conf,
            "active_random_amplification": finalized_metrics["active_random_amplification"]["true_class_confidence"] - original_conf,
        },
        "activation_summary": {
            "total_images": total_images,
            "ksae_k": ksae["k"],
            "amplification_factor": cfg["amplification_factor"],
            "total_active_features_seen": feature_counter["total_active_features"],
            "total_target_amplified": feature_counter["total_target_amplified"],
            "total_random_amplified": feature_counter["total_random_amplified"],
            "images_with_target_features": feature_counter["images_with_target_features"],
            "fraction_images_with_target_features": feature_counter["images_with_target_features"] / total_images if total_images else 0.0,
            "avg_target_amplified_per_image": feature_counter["total_target_amplified"] / total_images if total_images else 0.0,
            "avg_random_amplified_per_image": feature_counter["total_random_amplified"] / total_images if total_images else 0.0,
            "random_control": "active non-target-class features matched without replacement on absolute activation magnitude",
        },
        "amplified_images_summary": amplified_summary,
        "top_target_amplified_features": sorted_activation_dict(
            feature_counter["target_feature_counts"], feature_counter["target_feature_activation_sums"], stats
        ),
        "top_random_amplified_features": sorted_count_dict(feature_counter["random_feature_counts"]),
        "per_image_results": per_image_results,
    }


def make_summary_row(result):
    amplified = result["amplified_images_summary"]
    metrics = result["metrics"]
    return {
        "class_id": result["class_id"],
        "class_name": result["class_name"],
        "amplification_factor": result["amplification_factor"],
        "n_selected_features": result["selected_feature_count"],
        "n_test_images": metrics["original"]["total"],
        "n_amplified": amplified["total_amplified_images"],
        "fraction_amplified": amplified["fraction_amplified"],
        "mean_confidence_difference": amplified["targeted_minus_random_confidence_increase"],
        "mean_logit_difference": amplified["mean_targeted_minus_random_logit_increase"],
        "median_logit_difference": amplified["median_targeted_minus_random_logit_increase"],
        "mean_margin_difference": amplified["mean_targeted_minus_random_margin_increase"],
        "median_margin_difference": amplified["median_targeted_minus_random_margin_increase"],
        "ci_95_low": amplified["ci_95_low"],
        "ci_95_high": amplified["ci_95_high"],
        "positive_images": amplified["positive_images"],
        "negative_images": amplified["negative_images"],
        "fraction_positive": amplified["fraction_targeted_margin_increase_greater_than_random"],
        "wilcoxon_p": amplified["wilcoxon_p"],
        "sign_test_p": amplified["sign_test_p"],
        "original_accuracy": metrics["original"]["accuracy"],
        "targeted_accuracy": metrics["active_targeted_amplification"]["accuracy"],
        "random_accuracy": metrics["active_random_amplification"]["accuracy"],
    }


def skipped_result(class_id, class_name, target_feature_ids, n_test_images, amplification_factor):
    return {
        "class_id": class_id,
        "class_name": class_name,
        "amplification_factor": amplification_factor,
        "selected_feature_count": len(target_feature_ids),
        "selected_feature_ids": target_feature_ids,
        "metrics": {
            "original": {"accuracy": None, "total": n_test_images},
            "active_targeted_amplification": {"accuracy": None},
            "active_random_amplification": {"accuracy": None},
        },
        "amplified_images_summary": {
            "total_amplified_images": 0,
            "fraction_amplified": 0.0,
            "targeted_minus_random_confidence_increase": None,
            "mean_targeted_minus_random_logit_increase": None,
            "median_targeted_minus_random_logit_increase": None,
            "mean_targeted_minus_random_margin_increase": None,
            "median_targeted_minus_random_margin_increase": None,
            "ci_95_low": None,
            "ci_95_high": None,
            "positive_images": 0,
            "negative_images": 0,
            "fraction_targeted_margin_increase_greater_than_random": None,
            "wilcoxon_p": None,
            "sign_test_p": None,
        },
        "per_image_results": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)

    if cfg.get("amplification_factor", 1.5) <= 1.0:
        raise ValueError("amplification_factor must be > 1.0")
    cfg["amplification_factor"] = float(cfg.get("amplification_factor", 1.5))
    cfg["random_seed"] = cfg.get("random_seed", 42)

    sys.path.insert(0, cfg["diffc_dir"])
    sys.path.insert(1, cfg["sd_ksae_dir"])

    from helpers.dataset import HuggingFaceImageDataset, load_huggingface_dataset
    from constants import model_base_dict, diffusion_transformers_val, clip_transforms
    from models import ImageClassifer

    random.seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])
    torch.manual_seed(cfg["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Amplification factor:", cfg["amplification_factor"])

    print("Loading feature statistics...")
    stats = load_feature_stats(cfg["feature_dir"], device)
    feature_summary_df = pd.read_csv(cfg["feature_summary_path"])

    print("Loading k-SAE...")
    ksae = load_ksae(cfg["ksae_checkpoint_path"], device, default_k=cfg.get("ksae_k", 32))
    print("k-SAE k:", ksae["k"])
    print("k-SAE n_features:", ksae["n_features"])

    print("Loading dataset...")
    hf_test_dataset = load_huggingface_dataset(cfg["dataset_flag"], split=cfg.get("split", "test"))
    class_names = hf_test_dataset.features["label"].names
    labels_all = hf_test_dataset["label"]
    test_dataset = HuggingFaceImageDataset(hf_test_dataset, diffusion_transformers_val, clip_transforms)

    print("Loading DiffC model...")
    diffc_config = {
        "dataset_flag": cfg["dataset_flag"],
        "output_dir": "",
        "seed": cfg["random_seed"],
        "model_name": cfg["model_name"],
        "diffusion_timestep": cfg["diffusion_timestep"],
        "diffusion_layer": cfg["diffusion_layer"],
        "learning_rate": cfg.get("learning_rate", 1e-4),
        "num_epochs": cfg.get("num_epochs", 90),
        "batch_size": cfg["batch_size"],
        "prompt_type": cfg.get("prompt_type", "empty"),
        "pooling_strategy": cfg.get("pooling_strategy", "GAP"),
        "dropout_rate": cfg["dropout_rate"],
        "num_classes": cfg["num_classes"],
        "num_devices": 1,
        "feature_model": model_base_dict[cfg["model_name"]],
        "diffusion_step_type": cfg.get("diffusion_step_type", "onestep"),
        "device": device,
        "input_channels": cfg["input_channels"],
    }

    model = ImageClassifer(diffc_config).to(device)
    checkpoint = torch.load(cfg["diffc_checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded DiffC checkpoint epoch:", checkpoint.get("epoch"))

    target_classes_cfg = cfg.get("target_classes", "all")
    target_classes = list(range(len(class_names))) if target_classes_cfg == "all" else [int(x) for x in target_classes_cfg]
    for class_id in target_classes:
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"Invalid target class {class_id}. Valid range: 0..{len(class_names) - 1}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(cfg["output_dir"]) / f"run_{timestamp}_x{cfg['amplification_factor']:g}"
    details_dir = output_root / "per_class"
    details_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_rows = []

    for class_id in target_classes:
        class_name = class_names[class_id]
        target_feature_ids = select_target_features(
            feature_summary_df, class_id, cfg["min_purity"], cfg["min_valid"]
        )
        target_indices = [i for i, label in enumerate(labels_all) if int(label) == class_id][:cfg["max_images"]]

        print("\n" + "=" * 70)
        print(f"Class {class_id}: {class_name}")
        print(f"Selected features: {len(target_feature_ids)} | Test images: {len(target_indices)}")

        if not target_feature_ids or not target_indices:
            print("Skipping class: no selected features or no test images.")
            result = skipped_result(
                class_id, class_name, target_feature_ids, len(target_indices), cfg["amplification_factor"]
            )
        else:
            result = run_class(
                class_id, class_name, target_indices, target_feature_ids,
                test_dataset, model, ksae, stats, cfg, device,
            )

        detail_path = details_dir / f"class_{class_id:02d}_{class_name}.json"
        with detail_path.open("w") as f:
            json.dump(result, f, indent=2)

        all_results.append({
            "class_id": class_id,
            "class_name": class_name,
            "detail_file": str(detail_path),
            "amplified_images_summary": result["amplified_images_summary"],
        })
        summary_rows.append(make_summary_row(result))

    summary_df = pd.DataFrame(summary_rows).sort_values("class_id")
    summary_csv = output_root / "amplification_statistical_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    combined_json = output_root / "combined_summary.json"
    with combined_json.open("w") as f:
        json.dump({"run_timestamp": timestamp, "config": cfg, "classes": all_results}, f, indent=2)

    display_columns = [
        "class_id", "class_name", "amplification_factor", "n_selected_features", "n_amplified",
        "mean_margin_difference", "median_margin_difference", "ci_95_low", "ci_95_high",
        "fraction_positive", "wilcoxon_p", "sign_test_p",
    ]
    print("\n" + "=" * 70)
    print(summary_df[display_columns].to_string(index=False))
    print("\nSaved run to:", output_root)
    print("Summary CSV:", summary_csv)
    print("Combined JSON:", combined_json)


if __name__ == "__main__":
    main()
