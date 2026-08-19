import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

SD_KSAE_DIR = "/home/woody/rlvl/rlvl172v/revelio/SD-kSAE"
sys.path.insert(0, SD_KSAE_DIR)

from helpers.dataset import HuggingFaceImageDataset, load_huggingface_dataset
from constants import (
    model_base_dict,
    diffusion_transformers_val,
    clip_transforms,
)
from models import ImageClassifer


WORK = "/home/woody/rlvl/rlvl172v"

DIFFC_CHECKPOINT_PATH = (
    WORK
    + "/revelio/DiffC_outputs/timm-oxford-iiit-pet/runwayml-stable-diffusion-v1-5/"
    + "diffusion_step_25/layer_up_ft:1/prompt_empty/pool_GAP/dropout_0.0/"
    + "best_classifier.pt"
)

KSAE_CHECKPOINT_PATH = (
    WORK
    + "/revelio/SD-kSAE/Checkpoints/d0p5pu4r/"
    + "final_k_sparse_autoencoder_/home/woody/rlvl/rlvl172v/revelio/SD-kSAE/"
    + "oxfordpet/SDv1-5/timestep_25/up_blocks_1_10_up_blocks_1_81920.pt"
)

OUTPUT_PATH = (
    WORK
    + "/revelio/causal_results/persian_upft1_ablation.json"
)

TARGET_CLASS = 23  # Persian
TARGET_FEATURE_IDS = [16503, 3407, 59081]

MAX_IMAGES = 64
BATCH_SIZE = 8
RANDOM_SEED = 42


def load_ksae(device):
    checkpoint = torch.load(KSAE_CHECKPOINT_PATH, map_location=device)
    state = checkpoint["state_dict"]
    cfg = checkpoint["cfg"]

    ksae = {
        "W_enc": state["W_enc"].to(device),
        "b_enc": state["b_enc"].to(device),
        "W_dec": state["W_dec"].to(device),
        "b_dec": state["b_dec"].to(device),
        "k": cfg.k,
        "n_features": state["W_enc"].shape[1],
    }

    return ksae


def ksae_encode(x, ksae):
    W_enc = ksae["W_enc"]
    b_enc = ksae["b_enc"]
    b_dec = ksae["b_dec"]
    k = ksae["k"]

    pre_acts = (x - b_dec) @ W_enc + b_enc
    acts = torch.relu(pre_acts)

    top_values, top_indices = torch.topk(acts, k=k, dim=-1)

    sparse_acts = torch.zeros_like(acts)
    sparse_acts.scatter_(dim=-1, index=top_indices, src=top_values)

    return sparse_acts


def ksae_decode(sparse_acts, ksae):
    return sparse_acts @ ksae["W_dec"] + ksae["b_dec"]


def apply_ksae_intervention(features, ksae, ablate_ids=None):
    """
    features: [B, 1280, 32, 32]

    k-SAE was trained on pooled [B, 1280] vectors.
    We therefore modify the pooled channel vector and apply the resulting
    channel-wise delta back to the spatial feature map.
    """
    pooled = features.mean(dim=(2, 3))

    sparse_acts = ksae_encode(pooled, ksae)

    if ablate_ids is not None:
        sparse_acts = sparse_acts.clone()
        sparse_acts[:, ablate_ids] = 0.0

    reconstructed_pooled = ksae_decode(sparse_acts, ksae)

    delta = reconstructed_pooled - pooled
    modified_features = features + delta[:, :, None, None]

    return modified_features


def update_metrics(metrics, logits, labels, target_class):
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

    loss = F.cross_entropy(logits, labels, reduction="sum")

    metrics["loss_sum"] += loss.item()
    metrics["correct"] += (preds == labels).sum().item()
    metrics["total"] += labels.numel()
    metrics["true_conf_sum"] += probs[torch.arange(labels.numel(), device=labels.device), labels].sum().item()
    metrics["target_conf_sum"] += probs[:, target_class].sum().item()


def empty_metrics():
    return {
        "loss_sum": 0.0,
        "correct": 0,
        "total": 0,
        "true_conf_sum": 0.0,
        "target_conf_sum": 0.0,
    }


def finalize_metrics(metrics):
    total = metrics["total"]

    return {
        "accuracy": metrics["correct"] / total,
        "cross_entropy": metrics["loss_sum"] / total,
        "true_class_confidence": metrics["true_conf_sum"] / total,
        "target_class_confidence": metrics["target_conf_sum"] / total,
        "total": total,
    }


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "dataset_flag": "timm/oxford-iiit-pet",
        "output_dir": "",
        "seed": 42,
        "model_name": "runwayml/stable-diffusion-v1-5",
        "diffusion_timestep": 25,
        "diffusion_layer": "up_ft:1",
        "learning_rate": 1e-4,
        "num_epochs": 90,
        "batch_size": 16,
        "prompt_type": "empty",
        "pooling_strategy": "GAP",
        "dropout_rate": 0.0,
        "num_classes": 37,
        "num_devices": 1,
        "feature_model": model_base_dict["runwayml/stable-diffusion-v1-5"],
        "diffusion_step_type": "onestep",
        "device": device,
        "input_channels": 1280,
    }

    print("Loading k-SAE...")
    ksae = load_ksae(device)

    target_feature_ids = torch.tensor(
        TARGET_FEATURE_IDS,
        dtype=torch.long,
        device=device,
    )

    all_feature_ids = list(range(ksae["n_features"]))
    random_pool = [
        feature_id for feature_id in all_feature_ids
        if feature_id not in TARGET_FEATURE_IDS
    ]

    random_feature_ids = torch.tensor(
        random.sample(random_pool, len(TARGET_FEATURE_IDS)),
        dtype=torch.long,
        device=device,
    )

    print("Target class:", TARGET_CLASS, "Persian")
    print("Target feature IDs:", TARGET_FEATURE_IDS)
    print("Random feature IDs:", random_feature_ids.detach().cpu().tolist())

    print("Loading dataset...")
    hf_test_dataset = load_huggingface_dataset(
        config["dataset_flag"],
        split="test",
    )

    labels = hf_test_dataset["label"]
    target_indices = [
        i for i, label in enumerate(labels)
        if int(label) == TARGET_CLASS
    ][:MAX_IMAGES]

    print("Selected Persian test images:", len(target_indices))

    test_dataset = HuggingFaceImageDataset(
        hf_test_dataset,
        diffusion_transformers_val,
        clip_transforms,
    )

    subset = Subset(test_dataset, target_indices)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=False,
    )

    print("Loading DiffC model...")
    model = ImageClassifer(config).to(device)

    checkpoint = torch.load(DIFFC_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded DiffC checkpoint epoch:", checkpoint["epoch"])
    print("Loaded DiffC checkpoint test accuracy:", checkpoint["test_accuracy"])

    metrics = {
        "original": empty_metrics(),
        "sae_reconstruction": empty_metrics(),
        "targeted_ablation": empty_metrics(),
        "random_ablation": empty_metrics(),
    }

    target_active_counts = torch.zeros(len(TARGET_FEATURE_IDS), device=device)
    target_activation_sums = torch.zeros(len(TARGET_FEATURE_IDS), device=device)
    random_active_counts = torch.zeros(len(TARGET_FEATURE_IDS), device=device)
    random_activation_sums = torch.zeros(len(TARGET_FEATURE_IDS), device=device)

    with torch.no_grad():
        for diffusion_images, clip_images, labels, _ in loader:
            diffusion_images = diffusion_images.to(device)
            labels = labels.to(device)

            # Extract once only.
            features = model.get_features(
                diffusion_images,
                None,
                config["diffusion_timestep"],
            )

	    pooled = features.mean(dim=(2, 3))
	    sparse_acts = ksae_encode(pooled, ksae)

	    target_acts = sparse_acts[:, target_feature_ids]
	    random_acts = sparse_acts[:, random_feature_ids]

	    print("Target active counts per batch:", (target_acts > 0).sum(dim=0).detach().cpu().tolist())
	    print("Target mean activations:", target_acts.mean(dim=0).detach().cpu().tolist())
	    print("Random active counts per batch:", (random_acts > 0).sum(dim=0).detach().cpu().tolist())

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
                ablate_ids=target_feature_ids,
            )
            logits_target = model.classifer(features_target)

            features_random = apply_ksae_intervention(
                features,
                ksae,
                ablate_ids=random_feature_ids,
            )
            logits_random = model.classifer(features_random)

            update_metrics(
                metrics["original"],
                logits_original,
                labels,
                TARGET_CLASS,
            )
            update_metrics(
                metrics["sae_reconstruction"],
                logits_recon,
                labels,
                TARGET_CLASS,
            )
            update_metrics(
                metrics["targeted_ablation"],
                logits_target,
                labels,
                TARGET_CLASS,
            )
            update_metrics(
                metrics["random_ablation"],
                logits_random,
                labels,
                TARGET_CLASS,
            )

	    pooled = features.mean(dim=(2, 3))
	    sparse_acts = ksae_encode(pooled, ksae)

	    target_acts = sparse_acts[:, target_feature_ids]
	    random_acts = sparse_acts[:, random_feature_ids]

	    target_active_counts += (target_acts > 0).sum(dim=0)
	    target_activation_sums += target_acts.sum(dim=0)

	    random_active_counts += (random_acts > 0).sum(dim=0)
	    random_activation_sums += random_acts.sum(dim=0)

    results = {
        "target_class": TARGET_CLASS,
        "target_class_name": "persian",
        "target_feature_ids": TARGET_FEATURE_IDS,
        "random_feature_ids": random_feature_ids.detach().cpu().tolist(),
        "max_images": MAX_IMAGES,
        "batch_size": BATCH_SIZE,
        "metrics": {
            name: finalize_metrics(value)
            for name, value in metrics.items()
        },
    }

    original_conf = results["metrics"]["original"]["true_class_confidence"]
    target_conf = results["metrics"]["targeted_ablation"]["true_class_confidence"]
    random_conf = results["metrics"]["random_ablation"]["true_class_confidence"]

    results["confidence_drop"] = {
        "targeted_ablation": original_conf - target_conf,
        "random_ablation": original_conf - random_conf,
    }

    results["activation_stats"] = {
        "target_active_counts": target_active_counts.detach().cpu().tolist(),
        "target_activation_sums": target_activation_sums.detach().cpu().tolist(),
        "random_active_counts": random_active_counts.detach().cpu().tolist(),
        "random_activation_sums": random_activation_sums.detach().cpu().tolist(),
    }

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(results, f, indent=4)

    print()
    print(json.dumps(results, indent=4))
    print()
    print("Saved results to:", output_path)


if __name__ == "__main__":
    main()
