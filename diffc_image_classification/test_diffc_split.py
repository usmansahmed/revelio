import torch
from torch.utils.data import DataLoader, Subset

from helpers.dataset import HuggingFaceImageDataset, load_huggingface_dataset
from constants import (
    model_base_dict,
    diffusion_transformers_val,
    clip_transforms,
)
from models import ImageClassifer


WORK = "/home/woody/rlvl/rlvl172v"

CHECKPOINT_PATH = (
    WORK
    + "/revelio/DiffC_outputs/timm-oxford-iiit-pet/runwayml-stable-diffusion-v1-5/"
    + "diffusion_step_25/layer_up_ft:1/prompt_empty/pool_GAP/dropout_0.0/"
    + "best_classifier.pt"
)

TARGET_CLASS = 23
MAX_IMAGES = 16


def main():
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

    print("Loading dataset...")
    hf_test_dataset = load_huggingface_dataset(
        config["dataset_flag"],
        split="test",
    )

    labels = hf_test_dataset["label"]
    persian_indices = [
        i for i, label in enumerate(labels)
        if int(label) == TARGET_CLASS
    ][:MAX_IMAGES]

    print("Target class:", TARGET_CLASS)
    print("Selected images:", len(persian_indices))
    print("Indices:", persian_indices)

    test_dataset = HuggingFaceImageDataset(
        hf_test_dataset,
        diffusion_transformers_val,
        clip_transforms,
    )

    subset = Subset(test_dataset, persian_indices)

    loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=False,
        pin_memory=False,
    )

    print("Loading model...")
    model = ImageClassifer(config).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint test accuracy:", checkpoint["test_accuracy"])

    total = 0
    correct_full_forward = 0
    correct_split_forward = 0
    max_logit_diff = 0.0

    with torch.no_grad():
        for diffusion_images, clip_images, labels, _ in loader:
            diffusion_images = diffusion_images.to(device)
            labels = labels.to(device)

            logits_full = model(
                diffusion_images,
                None,
                config["diffusion_timestep"],
            )

            features = model.get_features(
                diffusion_images,
                None,
                config["diffusion_timestep"],
            )

            logits_split = model.classifer(features)

            diff = (logits_full - logits_split).abs().max().item()
            max_logit_diff = max(max_logit_diff, diff)

            pred_full = logits_full.argmax(dim=1)
            pred_split = logits_split.argmax(dim=1)

            correct_full_forward += (pred_full == labels).sum().item()
            correct_split_forward += (pred_split == labels).sum().item()
            total += labels.size(0)

            print("Feature shape:", tuple(features.shape))
            print("Labels:", labels.detach().cpu().tolist())
            print("Full preds:", pred_full.detach().cpu().tolist())
            print("Split preds:", pred_split.detach().cpu().tolist())
            print("True-class probs:", torch.softmax(logits_split, dim=1)[:, TARGET_CLASS].detach().cpu().tolist())

    print()
    print("Total images:", total)
    print("Full forward accuracy:", correct_full_forward / total)
    print("Split forward accuracy:", correct_split_forward / total)
    print("Max full-vs-split logit difference:", max_logit_diff)


if __name__ == "__main__":
    main()
