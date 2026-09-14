import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def eligible_reference_features(summary_df, min_purity, min_valid):
    selected = summary_df[
        (summary_df["label_purity"] >= min_purity)
        & (summary_df["valid_top_count"] >= min_valid)
    ]
    return selected["feature_id"].astype(int).tolist()


def compute_reference_activations(model, train_dataset, ksae, stats, eligible_ids, cfg, device):
    """Compute class-conditioned median non-zero activation for high-purity features."""
    cache_path = Path(cfg["reference_activation_cache"])
    if cache_path.exists() and not cfg.get("recompute_reference_activations", False):
        print("Loading cached reference activations:", cache_path)
        cache = torch.load(cache_path, map_location="cpu")
        return cache["median"].to(device), cache["count"].to(device)

    print("Computing training-set reference activations. This is done once and then cached...")
    eligible_mask = torch.zeros(ksae["n_features"], dtype=torch.bool, device=device)
    eligible_mask[torch.tensor(eligible_ids, dtype=torch.long, device=device)] = True
    activation_values = {int(fid): [] for fid in eligible_ids}

    indices = list(range(len(train_dataset)))
    max_per_class = cfg.get("max_reference_images_per_class")
    if max_per_class is not None:
        labels = train_dataset.dataset["label"] if hasattr(train_dataset, "dataset") else None
        if labels is not None:
            counts, subset = {}, []
            for idx, label in enumerate(labels):
                label = int(label)
                if counts.get(label, 0) < int(max_per_class):
                    subset.append(idx)
                    counts[label] = counts.get(label, 0) + 1
            indices = subset

    loader = DataLoader(
        Subset(train_dataset, indices),
        batch_size=cfg.get("reference_batch_size", cfg["batch_size"]),
        shuffle=False,
        pin_memory=False,
    )

    with torch.no_grad():
        for batch_idx, (diffusion_images, _, labels, _) in enumerate(loader):
            diffusion_images, labels = diffusion_images.to(device), labels.to(device)
            features = model.get_features(diffusion_images, None, cfg["diffusion_timestep"])
            sparse_acts = ksae_encode(features.mean(dim=(2, 3)), ksae)

            for i in range(labels.size(0)):
                label = int(labels[i].item())
                active_ids = torch.nonzero(sparse_acts[i] != 0, as_tuple=False).flatten()
                if active_ids.numel() == 0:
                    continue
                ids = active_ids[eligible_mask[active_ids] & (stats["majority_label"][active_ids] == label)]
                for fid in ids.detach().cpu().tolist():
                    activation_values[int(fid)].append(float(sparse_acts[i, fid].item()))

            if batch_idx % 25 == 0:
                print(f"  reference batch {batch_idx}/{len(loader)}")

    median = torch.full((ksae["n_features"],), float("nan"), dtype=torch.float32)
    count = torch.zeros(ksae["n_features"], dtype=torch.long)
    for fid, values in activation_values.items():
        if values:
            median[fid] = float(np.median(np.asarray(values, dtype=np.float32)))
            count[fid] = len(values)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"median": median, "count": count}, cache_path)
    print("Saved reference activation cache:", cache_path)
    return median.to(device), count.to(device)


def score_inactive_target_features(feature_ids, reference_median, stats, ranking_method):
    ids = torch.tensor(feature_ids, dtype=torch.long, device=reference_median.device)
    ref = reference_median[ids].abs()
    purity = stats["label_purity"][ids]
    sparsity = stats["sparsity"][ids]

    if ranking_method == "reference_activation":
        score = ref
    elif ranking_method == "purity":
        score = purity
    elif ranking_method == "purity_reference_activation":
        score = purity * ref
    elif ranking_method == "sparse_class_specific":
        score = purity * ref * sparsity
    else:
        raise ValueError(f"Unknown insertion_feature_ranking: {ranking_method}")

    return ids[torch.argsort(score, descending=True)]


def choose_control_features(target_ids, active_ids, target_class, control_pool_ids, reference_median, decoder_norms, stats):
    """Match unrelated inactive controls on reference activation, decoder norm and purity."""
    if target_ids.numel() == 0:
        return target_ids

    active_mask = torch.zeros(reference_median.numel(), dtype=torch.bool, device=reference_median.device)
    active_mask[active_ids] = True
    control_pool = control_pool_ids[~active_mask[control_pool_ids]]
    control_pool = control_pool[stats["majority_label"][control_pool] != target_class]

    matched, available = [], control_pool
    eps = 1e-8
    for target_id in target_ids:
        if available.numel() == 0:
            break

        target_ref = reference_median[target_id].abs().clamp_min(eps)
        control_ref = reference_median[available].abs().clamp_min(eps)
        target_norm = decoder_norms[target_id].clamp_min(eps)
        control_norm = decoder_norms[available].clamp_min(eps)

        activation_distance = torch.abs(torch.log(control_ref / target_ref))
        decoder_distance = torch.abs(torch.log(control_norm / target_norm))
        purity_distance = torch.abs(stats["label_purity"][available] - stats["label_purity"][target_id])
        distance = activation_distance + decoder_distance + 0.25 * purity_distance
        distance = distance + torch.rand_like(distance) * 1e-12

        pos = torch.argmin(distance)
        matched.append(available[pos])
        available = torch.cat((available[:pos], available[pos + 1:]))

    return torch.stack(matched) if matched else control_pool[:0]


def choose_source_removals(sparse_row, active_ids, target_class, stats, n_remove):
    """Remove the weakest active non-target features to keep exactly k active features."""
    pool = active_ids[stats["majority_label"][active_ids] != target_class]
    if pool.numel() == 0 or n_remove == 0:
        return pool[:0]
    order = torch.argsort(sparse_row[pool].abs(), descending=False)
    return pool[order[:min(n_remove, pool.numel())]]


def target_margin(logits, target_class):
    target_logit = logits[:, target_class]
    others = logits.clone()
    others[:, target_class] = -torch.inf
    return target_logit - others.max(dim=1).values


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


def benjamini_hochberg(p_values):
    adjusted = [None] * len(p_values)
    valid = [(i, float(p)) for i, p in enumerate(p_values) if p is not None and not pd.isna(p)]
    if not valid:
        return adjusted

    valid.sort(key=lambda x: x[1])
    m = len(valid)
    raw = [[idx, min(p * m / rank, 1.0)] for rank, (idx, p) in enumerate(valid, start=1)]

    running_min = 1.0
    for i in range(len(raw) - 1, -1, -1):
        idx, value = raw[i]
        running_min = min(running_min, value)
        adjusted[idx] = running_min
    return adjusted


def summarize_insertions(rows, bootstrap_samples, seed):
    if not rows:
        return {
            "n_source_images": 0,
            "mean_original_target_margin": None,
            "mean_targeted_target_margin": None,
            "mean_control_target_margin": None,
            "mean_targeted_margin_increase": None,
            "mean_control_margin_increase": None,
            "mean_target_margin_difference": None,
            "median_target_margin_difference": None,
            "ci_95_low": None,
            "ci_95_high": None,
            "positive_images": 0,
            "negative_images": 0,
            "fraction_positive": None,
            "wilcoxon_p": None,
            "sign_test_p": None,
            "targeted_flip_count": 0,
            "control_flip_count": 0,
            "targeted_crossed_zero_count": 0,
            "control_crossed_zero_count": 0,
        }

    original = np.array([r["original_target_margin"] for r in rows], dtype=float)
    targeted = np.array([r["targeted_target_margin"] for r in rows], dtype=float)
    control = np.array([r["control_target_margin"] for r in rows], dtype=float)
    targeted_inc = targeted - original
    control_inc = control - original
    difference = targeted - control

    ci_low, ci_high = bootstrap_ci(difference, n_bootstrap=bootstrap_samples, seed=seed)
    wilcoxon_p, sign_p, positive, negative, fraction_positive = paired_tests(difference)
    targeted_flips = sum(r["targeted_pred"] == r["target_class"] for r in rows)
    control_flips = sum(r["control_pred"] == r["target_class"] for r in rows)
    targeted_cross = int(np.sum((original <= 0) & (targeted > 0)))
    control_cross = int(np.sum((original <= 0) & (control > 0)))

    return {
        "n_source_images": len(rows),
        "mean_original_target_margin": float(original.mean()),
        "mean_targeted_target_margin": float(targeted.mean()),
        "mean_control_target_margin": float(control.mean()),
        "mean_targeted_margin_increase": float(targeted_inc.mean()),
        "mean_control_margin_increase": float(control_inc.mean()),
        "mean_target_margin_difference": float(difference.mean()),
        "median_target_margin_difference": float(np.median(difference)),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "positive_images": positive,
        "negative_images": negative,
        "fraction_positive": fraction_positive,
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
        "targeted_flip_count": int(targeted_flips),
        "control_flip_count": int(control_flips),
        "targeted_crossed_zero_count": targeted_cross,
        "control_crossed_zero_count": control_cross,
    }


def run_target_class_dose_response(
    target_class, target_feature_ids, candidate_indices, test_dataset, model, ksae, stats,
    reference_median, reference_count, control_pool_ids, decoder_norms, cfg, device,
):
    """Run all insertion scales using identical images and identical feature swaps."""
    rng = random.Random(cfg["random_seed"] + target_class)
    candidate_indices = list(candidate_indices)
    rng.shuffle(candidate_indices)

    loader = DataLoader(
        Subset(test_dataset, candidate_indices),
        batch_size=cfg["batch_size"],
        shuffle=False,
        pin_memory=False,
    )

    target_feature_ids = [
        fid for fid in target_feature_ids
        if int(reference_count[fid].item()) >= cfg.get("min_reference_count", 5)
        and not torch.isnan(reference_median[fid])
    ]
    if not target_feature_ids:
        return {}, {"eligible_target_features": [], "selected_interventions": 0}

    ranked_target_ids = score_inactive_target_features(
        target_feature_ids, reference_median, stats,
        cfg.get("insertion_feature_ranking", "purity_reference_activation"),
    )
    target_feature_set = torch.zeros(ksae["n_features"], dtype=torch.bool, device=device)
    target_feature_set[ranked_target_ids] = True

    scales = [float(x) for x in cfg["insertion_scales"]]
    results_by_scale = {scale: [] for scale in scales}
    max_source_images = int(cfg.get("max_source_images_per_target", 64))
    max_insert = int(cfg.get("max_insert_per_image", 3))
    selected_source_count = 0
    sample_offset = 0

    torch.manual_seed(cfg["random_seed"] + target_class)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["random_seed"] + target_class)

    with torch.no_grad():
        for batch_idx, (diffusion_images, _, labels, _) in enumerate(loader):
            if selected_source_count >= max_source_images:
                break

            diffusion_images, labels = diffusion_images.to(device), labels.to(device)
            features = model.get_features(diffusion_images, None, cfg["diffusion_timestep"])
            pooled = features.mean(dim=(2, 3))
            sparse_acts = ksae_encode(pooled, ksae)
            recon_pooled = ksae_decode(sparse_acts, ksae)
            logits_original = model.classifer(features)
            probs_original = torch.softmax(logits_original, dim=1)
            margin_original = target_margin(logits_original, target_class)
            batch_dataset_indices = candidate_indices[sample_offset:sample_offset + labels.size(0)]

            interventions = [None] * labels.size(0)
            remaining = max_source_images - selected_source_count
            selected_in_batch = 0

            for i in range(labels.size(0)):
                if selected_in_batch >= remaining:
                    break

                true_label = int(labels[i].item())
                original_pred = int(logits_original[i].argmax().item())
                if true_label == target_class or original_pred == target_class:
                    continue
                if cfg.get("require_originally_correct", True) and original_pred != true_label:
                    continue

                active_ids = torch.nonzero(sparse_acts[i] != 0, as_tuple=False).flatten()
                if active_ids.numel() == 0:
                    continue

                active_target_ids = active_ids[target_feature_set[active_ids]]
                if cfg.get("require_no_target_features_active", True) and active_target_ids.numel() > 0:
                    continue

                inactive_target_ids = ranked_target_ids[~torch.isin(ranked_target_ids, active_ids)]
                if inactive_target_ids.numel() == 0:
                    continue
                target_ids = inactive_target_ids[:max_insert]

                control_ids = choose_control_features(
                    target_ids, active_ids, target_class, control_pool_ids,
                    reference_median, decoder_norms, stats,
                )
                n_pairs = min(target_ids.numel(), control_ids.numel())
                if n_pairs == 0:
                    continue

                target_ids = target_ids[:n_pairs]
                control_ids = control_ids[:n_pairs]
                remove_ids = choose_source_removals(sparse_acts[i], active_ids, target_class, stats, n_pairs)
                n_final = min(n_pairs, remove_ids.numel())
                if n_final == 0:
                    continue

                target_ids = target_ids[:n_final]
                control_ids = control_ids[:n_final]
                remove_ids = remove_ids[:n_final]

                interventions[i] = {
                    "image_index": int(batch_dataset_indices[i]),
                    "true_label": true_label,
                    "original_pred": original_pred,
                    "target_neuron_ids": target_ids,
                    "control_neuron_ids": control_ids,
                    "removed_source_neuron_ids": remove_ids,
                    "target_reference_activations": reference_median[target_ids].clone(),
                    "control_reference_activations": reference_median[control_ids].clone(),
                    "removed_source_activations": sparse_acts[i, remove_ids].clone(),
                    "number_features_inserted": int(n_final),
                }
                selected_in_batch += 1

            if selected_in_batch == 0:
                sample_offset += labels.size(0)
                continue

            for scale in scales:
                sparse_target = sparse_acts.clone()
                sparse_control = sparse_acts.clone()

                for i, intervention in enumerate(interventions):
                    if intervention is None:
                        continue

                    target_ids = intervention["target_neuron_ids"]
                    control_ids = intervention["control_neuron_ids"]
                    remove_ids = intervention["removed_source_neuron_ids"]

                    sparse_target[i, remove_ids] = 0.0
                    sparse_control[i, remove_ids] = 0.0
                    sparse_target[i, target_ids] = intervention["target_reference_activations"] * scale
                    sparse_control[i, control_ids] = intervention["control_reference_activations"] * scale

                target_pooled = ksae_decode(sparse_target, ksae)
                control_pooled = ksae_decode(sparse_control, ksae)
                target_features = features + (target_pooled - recon_pooled)[:, :, None, None]
                control_features = features + (control_pooled - recon_pooled)[:, :, None, None]

                logits_target = model.classifer(target_features)
                logits_control = model.classifer(control_features)
                probs_target = torch.softmax(logits_target, dim=1)
                probs_control = torch.softmax(logits_control, dim=1)
                margin_target = target_margin(logits_target, target_class)
                margin_control = target_margin(logits_control, target_class)

                for i, intervention in enumerate(interventions):
                    if intervention is None:
                        continue

                    original_conf = float(probs_original[i, target_class].item())
                    targeted_conf = float(probs_target[i, target_class].item())
                    control_conf = float(probs_control[i, target_class].item())
                    original_logit = float(logits_original[i, target_class].item())
                    targeted_logit = float(logits_target[i, target_class].item())
                    control_logit = float(logits_control[i, target_class].item())
                    original_margin = float(margin_original[i].item())
                    targeted_margin = float(margin_target[i].item())
                    control_margin = float(margin_control[i].item())

                    results_by_scale[scale].append({
                        "image_index": intervention["image_index"],
                        "true_label": intervention["true_label"],
                        "target_class": target_class,
                        "insertion_scale": scale,
                        "original_pred": intervention["original_pred"],
                        "targeted_pred": int(logits_target[i].argmax().item()),
                        "control_pred": int(logits_control[i].argmax().item()),
                        "original_target_confidence": original_conf,
                        "targeted_target_confidence": targeted_conf,
                        "control_target_confidence": control_conf,
                        "target_confidence_increase_difference": (targeted_conf - original_conf) - (control_conf - original_conf),
                        "original_target_logit": original_logit,
                        "targeted_target_logit": targeted_logit,
                        "control_target_logit": control_logit,
                        "target_logit_increase_difference": (targeted_logit - original_logit) - (control_logit - original_logit),
                        "original_target_margin": original_margin,
                        "targeted_target_margin": targeted_margin,
                        "control_target_margin": control_margin,
                        "targeted_target_margin_increase": targeted_margin - original_margin,
                        "control_target_margin_increase": control_margin - original_margin,
                        "target_margin_increase_difference": targeted_margin - control_margin,
                        "target_neuron_ids": [int(x) for x in intervention["target_neuron_ids"].detach().cpu().tolist()],
                        "control_neuron_ids": [int(x) for x in intervention["control_neuron_ids"].detach().cpu().tolist()],
                        "removed_source_neuron_ids": [int(x) for x in intervention["removed_source_neuron_ids"].detach().cpu().tolist()],
                        "target_inserted_activations": [
                            float(x) for x in (intervention["target_reference_activations"] * scale).detach().cpu().tolist()
                        ],
                        "control_inserted_activations": [
                            float(x) for x in (intervention["control_reference_activations"] * scale).detach().cpu().tolist()
                        ],
                        "removed_source_activations": [
                            float(x) for x in intervention["removed_source_activations"].detach().cpu().tolist()
                        ],
                        "number_features_inserted": intervention["number_features_inserted"],
                    })

            selected_source_count += selected_in_batch
            sample_offset += labels.size(0)
            print(f"  target {target_class} batch {batch_idx}: selected {selected_source_count}/{max_source_images}")

    metadata = {
        "eligible_target_features": [int(x) for x in ranked_target_ids.detach().cpu().tolist()],
        "reference_counts": {str(int(fid)): int(reference_count[fid].item()) for fid in ranked_target_ids},
        "reference_medians": {str(int(fid)): float(reference_median[fid].item()) for fid in ranked_target_ids},
        "selected_interventions": selected_source_count,
        "insertion_scales": scales,
    }
    return results_by_scale, metadata


def build_diffc_model(cfg, device):
    from constants import model_base_dict
    from models import ImageClassifer

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
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    cfg["random_seed"] = int(cfg.get("random_seed", 42))
    if "insertion_scales" not in cfg or not cfg["insertion_scales"]:
        raise ValueError("Config must contain a non-empty 'insertion_scales' list.")

    sys.path.insert(0, cfg["diffc_dir"])
    sys.path.insert(1, cfg["sd_ksae_dir"])

    from helpers.dataset import HuggingFaceImageDataset, load_huggingface_dataset
    from constants import diffusion_transformers_val, clip_transforms

    random.seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])
    torch.manual_seed(cfg["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    stats = load_feature_stats(cfg["feature_dir"], device)
    feature_summary_df = pd.read_csv(cfg["feature_summary_path"])
    ksae = load_ksae(cfg["ksae_checkpoint_path"], device, default_k=cfg.get("ksae_k", 32))
    decoder_norms = torch.linalg.vector_norm(ksae["W_dec"], dim=1)
    print("k-SAE k:", ksae["k"], "| n_features:", ksae["n_features"])

    model = build_diffc_model(cfg, device)

    print("Loading train/test datasets...")
    hf_train = load_huggingface_dataset(cfg["dataset_flag"], split=cfg.get("reference_split", "train"))
    hf_test = load_huggingface_dataset(cfg["dataset_flag"], split=cfg.get("test_split", "test"))
    class_names = hf_test.features["label"].names
    test_labels = [int(x) for x in hf_test["label"]]
    train_dataset = HuggingFaceImageDataset(hf_train, diffusion_transformers_val, clip_transforms)
    test_dataset = HuggingFaceImageDataset(hf_test, diffusion_transformers_val, clip_transforms)

    reference_feature_ids = eligible_reference_features(
        feature_summary_df,
        cfg.get("control_min_purity", cfg["min_purity"]),
        cfg.get("control_min_valid", cfg["min_valid"]),
    )
    reference_median, reference_count = compute_reference_activations(
        model, train_dataset, ksae, stats, reference_feature_ids, cfg, device
    )

    min_reference_count = int(cfg.get("min_reference_count", 5))
    control_ids = [
        fid for fid in reference_feature_ids
        if int(reference_count[fid].item()) >= min_reference_count and not torch.isnan(reference_median[fid])
    ]
    control_pool_ids = torch.tensor(control_ids, dtype=torch.long, device=device)

    target_classes_cfg = cfg.get("target_classes", "all")
    target_classes = list(range(len(class_names))) if target_classes_cfg == "all" else [int(x) for x in target_classes_cfg]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(cfg["output_dir"]) / f"run_{timestamp}"
    detail_dir = output_root / "per_target_class"
    detail_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    combined = []

    for target_class in target_classes:
        target_name = class_names[target_class]
        target_feature_ids = select_target_features(feature_summary_df, target_class, cfg["min_purity"], cfg["min_valid"])
        candidate_indices = [i for i, label in enumerate(test_labels) if label != target_class]

        print("\n" + "=" * 72)
        print(f"Target class {target_class}: {target_name}")
        print(f"High-purity target features: {len(target_feature_ids)} | candidate non-target images: {len(candidate_indices)}")

        if not target_feature_ids:
            results_by_scale = {float(s): [] for s in cfg["insertion_scales"]}
            metadata = {"eligible_target_features": [], "selected_interventions": 0}
        else:
            results_by_scale, metadata = run_target_class_dose_response(
                target_class, target_feature_ids, candidate_indices, test_dataset, model, ksae, stats,
                reference_median, reference_count, control_pool_ids, decoder_norms, cfg, device,
            )

        scale_details = {}
        for scale in [float(x) for x in cfg["insertion_scales"]]:
            rows = results_by_scale.get(scale, [])
            summary = summarize_insertions(
                rows,
                bootstrap_samples=int(cfg.get("bootstrap_samples", 10000)),
                seed=cfg["random_seed"] + target_class + int(round(scale * 1000)),
            )

            scale_key = str(scale)
            scale_details[scale_key] = {"summary": summary, "per_image_results": rows}
            summary_rows.append({
                "target_class": target_class,
                "target_class_name": target_name,
                "insertion_scale": scale,
                "n_selected_features": len(target_feature_ids),
                **summary,
            })

        detail = {
            "target_class": target_class,
            "target_class_name": target_name,
            "selected_feature_count": len(target_feature_ids),
            "selected_feature_ids": target_feature_ids,
            "insertion_scales": [float(x) for x in cfg["insertion_scales"]],
            "max_insert_per_image": int(cfg.get("max_insert_per_image", 3)),
            "metadata": metadata,
            "dose_response": scale_details,
        }

        detail_path = detail_dir / f"target_{target_class:02d}_{target_name}.json"
        with detail_path.open("w") as f:
            json.dump(detail, f, indent=2)

        combined.append({
            "target_class": target_class,
            "target_class_name": target_name,
            "detail_file": str(detail_path),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["insertion_scale", "target_class"]).reset_index(drop=True)

    # FDR within each dose, plus a more conservative global correction across all class x dose tests.
    summary_df["wilcoxon_fdr_within_scale"] = np.nan
    for scale in summary_df["insertion_scale"].unique():
        mask = summary_df["insertion_scale"] == scale
        summary_df.loc[mask, "wilcoxon_fdr_within_scale"] = benjamini_hochberg(
            summary_df.loc[mask, "wilcoxon_p"].tolist()
        )

    summary_df["wilcoxon_fdr_global"] = benjamini_hochberg(summary_df["wilcoxon_p"].tolist())

    summary_csv = output_root / "sufficiency_dose_response_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    with (output_root / "combined_summary.json").open("w") as f:
        json.dump({"run_timestamp": timestamp, "config": cfg, "targets": combined}, f, indent=2)

    display_cols = [
        "target_class", "target_class_name", "insertion_scale", "n_source_images",
        "mean_original_target_margin", "mean_targeted_target_margin", "mean_control_target_margin",
        "mean_targeted_margin_increase", "mean_control_margin_increase", "mean_target_margin_difference",
        "targeted_crossed_zero_count", "control_crossed_zero_count",
        "fraction_positive", "wilcoxon_fdr_within_scale",
    ]
    print("\n" + "=" * 72)
    print(summary_df[display_cols].to_string(index=False))
    print("\nSaved run to:", output_root)
    print("Summary CSV:", summary_csv)


if __name__ == "__main__":
    main()
