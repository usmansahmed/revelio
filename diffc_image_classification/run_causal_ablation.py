import argparse
import json
import random
import sys
from pathlib import Path

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


def load_feature_stats(feature_dir, device="cpu"):
    feature_dir = Path(feature_dir)

    return {
        "label_purity": load_tensor(feature_dir / "label_purity_top10.pt", device),
        "majority_label": load_tensor(feature_dir / "majority_label_top10.pt", device),
        "valid_count": load_tensor(feature_dir / "valid_top_count_top10.pt", device),
        "mean_acts": load_tensor(feature_dir / "sae_mean_acts.pt", device),
        "sparsity": load_tensor(feature_dir / "sae_sparsity.pt", device),
    }


def rank_candidate_features(stats, target_class, min_purity, min_valid, top_n, ranking_method):
    label_purity = stats["label_purity"]
    majority_label = stats["majority_label"]
    valid_count = stats["valid_count"]
    mean_acts = stats["mean_acts"]
    sparsity = stats["sparsity"]

    mask = (
        (majority_label == target_class)
        & (label_purity >= min_purity)
        & (valid_count >= min_valid)
    )

    feature_ids = torch.nonzero(mask, as_tuple=False).flatten()

    if len(feature_ids) == 0:
        raise RuntimeError(
            "No matching features found. Try lowering min_purity or min_valid."
        )

    if ranking_method == "purity_mean_activation":
        scores = label_purity[feature_ids] * mean_acts[feature_ids]
    elif ranking_method == "sparse_class_specific":
        scores = label_purity[feature_ids] * mean_acts[feature_ids] * sparsity[feature_ids]
    elif ranking_method == "sparsity":
        scores = sparsity[feature_ids]
    elif ranking_method == "mean_activation":
        scores = mean_acts[feature_ids]
    elif ranking_method == "purity":
        scores = label_purity[feature_ids]
    else:
        raise ValueError(f"Unknown ranking method: {ranking_method}")

    order = torch.argsort(scores, descending=True)
    selected = feature_ids[order[:top_n]]

    candidate_rows = []

    for fid in selected.tolist():
        candidate_rows.append({
            "feature_id": int(fid),
            "purity": float(label_purity[fid].item()),
            "valid_count": int(valid_count[fid].item()),
            "mean_activation": float(mean_acts[fid].item()),
            "sparsity": float(sparsity[fid].item()),
            "score": float(scores[order][len(candidate_rows)].item()),
        })

    return selected.tolist(), candidate_rows, int(len(feature_ids))


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
    pre_acts = (x - ksae["b_dec"]) @ ksae["W_enc"] + ksae["b_enc"]
    acts = torch.relu(pre_acts)

    top_values, top_indices = torch.topk(acts, k=ksae["k"], dim=-1)

    sparse_acts = torch.zeros_like(acts)
    sparse_acts.scatter_(dim=-1, index=top_indices, src=top_values)

    return sparse_acts


def ksae_decode(sparse_acts, ksae):
    return sparse_acts @ ksae["W_dec"] + ksae["b_dec"]


def apply_ksae_intervention(features, ksae, ablate_ids=None):
    pooled = features.mean(dim=(2, 3))
    sparse_acts = ksae_encode(pooled, ksae)

    if ablate_ids is not None:
        sparse_acts = sparse_acts.clone()
        sparse_acts[:, ablate_ids] = 0.0

    reconstructed_pooled = ksae_decode(sparse_acts, ksae)

    delta = reconstructed_pooled - pooled
    modified_features = features + delta[:, :, None, None]

    return modified_features


def empty_metrics():
    return {
        "loss_sum": 0.0,
        "correct": 0,
        "total": 0,
        "true_conf_sum": 0.0,
        "target_conf_sum": 0.0,
    }


def update_metrics(metrics, logits, labels, target_class):
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

    metrics["loss_sum"] += F.cross_entropy(logits, labels, reduction="sum").item()
    metrics["correct"] += (preds == labels).sum().item()
    metrics["total"] += labels.numel()
    metrics["true_conf_sum"] += probs[
        torch.arange(labels.numel(), device=labels.device),
        labels,
    ].sum().item()
    metrics["target_conf_sum"] += probs[:, target_class].sum().item()


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
        "accuracy": metrics["correct"] / total,
        "cross_entropy": metrics["loss_sum"] / total,
        "true_class_confidence": metrics["true_conf_sum"] / total,
        "target_class_confidence": metrics["target_conf_sum"] / total,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)

    sd_ksae_dir = cfg["sd_ksae_dir"]
    sys.path.insert(0, sd_ksae_dir)

    from helpers.dataset import HuggingFaceImageDataset, load_huggingface_dataset
    from constants import model_base_dict, diffusion_transformers_val, clip_transforms
    from models import ImageClassifer

    random_seed = cfg.get("random_seed", 42)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Target class:", cfg["target_class"], cfg.get("target_class_name", ""))

    print()
    print("Loading feature statistics...")
    stats_cpu = load_feature_stats(cfg["feature_dir"], device="cpu")

    target_feature_ids, candidate_rows, total_matching_features = rank_candidate_features(
        stats_cpu,
        target_class=cfg["target_class"],
        min_purity=cfg["min_purity"],
        min_valid=cfg["min_valid"],
        top_n=cfg["top_n"],
        ranking_method=cfg["ranking_method"],
    )

    print("Total matching features:", total_matching_features)
    print("Selected target features:", target_feature_ids)
    print()
    print("Selected feature details:")
    for row in candidate_rows:
        print(row)

    print()
    print("Loading k-SAE...")
    ksae = load_ksae(
        cfg["ksae_checkpoint_path"],
        device=device,
        default_k=cfg.get("ksae_k", 32),
    )

    target_feature_ids_tensor = torch.tensor(
        target_feature_ids,
        dtype=torch.long,
        device=device,
    )

    random_pool = [
        feature_id for feature_id in range(ksae["n_features"])
        if feature_id not in target_feature_ids
    ]

    random_feature_ids = random.sample(random_pool, len(target_feature_ids))
    random_feature_ids_tensor = torch.tensor(
        random_feature_ids,
        dtype=torch.long,
        device=device,
    )

    print("Random feature IDs:", random_feature_ids)

    print()
    print("Loading dataset...")
    hf_test_dataset = load_huggingface_dataset(
        cfg["dataset_flag"],
        split=cfg.get("split", "test"),
    )

    labels_all = hf_test_dataset["label"]

    target_indices = [
        i for i, label in enumerate(labels_all)
        if int(label) == cfg["target_class"]
    ][:cfg["max_images"]]

    if len(target_indices) == 0:
        raise RuntimeError(
            f"No images found for target class {cfg['target_class']}."
        )

    print("Selected target-class test images:", len(target_indices))
    print("First indices:", target_indices[:20])

    test_dataset = HuggingFaceImageDataset(
        hf_test_dataset,
        diffusion_transformers_val,
        clip_transforms,
    )

    loader = DataLoader(
        Subset(test_dataset, target_indices),
        batch_size=cfg["batch_size"],
        shuffle=False,
        pin_memory=False,
    )

    print()
    print("Loading DiffC model...")

    diffc_config = {
        "dataset_flag": cfg["dataset_flag"],
        "output_dir": "",
        "seed": random_seed,
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
    print("Loaded DiffC checkpoint test accuracy:", checkpoint.get("test_accuracy"))

    metrics = {
        "original": empty_metrics(),
        "sae_reconstruction": empty_metrics(),
        "targeted_ablation": empty_metrics(),
        "random_ablation": empty_metrics(),
    }

    target_active_counts = torch.zeros(len(target_feature_ids), device=device)
    target_activation_sums = torch.zeros(len(target_feature_ids), device=device)
    random_active_counts = torch.zeros(len(random_feature_ids), device=device)
    random_activation_sums = torch.zeros(len(random_feature_ids), device=device)

    with torch.no_grad():
        for batch_idx, (diffusion_images, clip_images, labels, _) in enumerate(loader):
            diffusion_images = diffusion_images.to(device)
            labels = labels.to(device)

            features = model.get_features(
                diffusion_images,
                None,
                cfg["diffusion_timestep"],
            )

            pooled = features.mean(dim=(2, 3))
            sparse_acts = ksae_encode(pooled, ksae)

            target_acts = sparse_acts[:, target_feature_ids_tensor]
            random_acts = sparse_acts[:, random_feature_ids_tensor]

            target_active_counts += (target_acts > 0).sum(dim=0)
            target_activation_sums += target_acts.sum(dim=0)

            random_active_counts += (random_acts > 0).sum(dim=0)
            random_activation_sums += random_acts.sum(dim=0)

            logits_original = model.classifer(features)

            features_recon = apply_ksae_intervention(
                features,
                ksae,
                ablate_ids=None,
            )
            logits_recon = model.classifer(features_recon)

            features_target = apply_ksae_intervention(
                features,
                ksae,
                ablate_ids=target_feature_ids_tensor,
            )
            logits_target = model.classifer(features_target)

            features_random = apply_ksae_intervention(
                features,
                ksae,
                ablate_ids=random_feature_ids_tensor,
            )
            logits_random = model.classifer(features_random)

            update_metrics(metrics["original"], logits_original, labels, cfg["target_class"])
            update_metrics(metrics["sae_reconstruction"], logits_recon, labels, cfg["target_class"])
            update_metrics(metrics["targeted_ablation"], logits_target, labels, cfg["target_class"])
            update_metrics(metrics["random_ablation"], logits_random, labels, cfg["target_class"])

            print(
                f"Batch {batch_idx}: "
                f"target active total={(target_acts > 0).sum().item()}, "
                f"random active total={(random_acts > 0).sum().item()}"
            )

    finalized_metrics = {
        name: finalize_metrics(value)
        for name, value in metrics.items()
    }

    original_conf = finalized_metrics["original"]["true_class_confidence"]
    recon_conf = finalized_metrics["sae_reconstruction"]["true_class_confidence"]
    target_conf = finalized_metrics["targeted_ablation"]["true_class_confidence"]
    random_conf = finalized_metrics["random_ablation"]["true_class_confidence"]

    results = {
        "config": cfg,
        "selected_features": candidate_rows,
        "target_feature_ids": target_feature_ids,
        "random_feature_ids": random_feature_ids,
        "total_matching_features": total_matching_features,
        "metrics": finalized_metrics,
        "confidence_drop_from_original": {
            "sae_reconstruction": original_conf - recon_conf,
            "targeted_ablation": original_conf - target_conf,
            "random_ablation": original_conf - random_conf,
        },
        "extra_drop_after_reconstruction": {
            "targeted_minus_reconstruction": recon_conf - target_conf,
            "random_minus_reconstruction": recon_conf - random_conf,
        },
        "activation_stats": {
            "target_active_counts": target_active_counts.detach().cpu().tolist(),
            "target_activation_sums": target_activation_sums.detach().cpu().tolist(),
            "random_active_counts": random_active_counts.detach().cpu().tolist(),
            "random_activation_sums": random_activation_sums.detach().cpu().tolist(),
        },
    }

    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(results, f, indent=4)

    print()
    print(json.dumps(results, indent=4))
    print()
    print("Saved results to:", output_path)


if __name__ == "__main__":
    main()