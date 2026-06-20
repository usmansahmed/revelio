from pathlib import Path
from collections import Counter

import torch
import pandas as pd

from config import DATASET_LAYER_DIRS, TOP_K_FOR_PURITY


def safe_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.load(path, map_location="cpu")


def compute_stats_for_layer(layer_name: str, feature_dir: Path, top_k: int):
    print(f"\nProcessing layer: {layer_name}")
    print(f"Feature directory: {feature_dir}")

    labels_path = feature_dir / "max_activating_image_label_indices.pt"
    values_path = feature_dir / "max_activating_image_values.pt"
    indices_path = feature_dir / "max_activating_image_indices.pt"
    mean_acts_path = feature_dir / "sae_mean_acts.pt"
    sparsity_path = feature_dir / "sae_sparsity.pt"

    labels = safe_load(labels_path)
    values = safe_load(values_path)
    indices = safe_load(indices_path)
    mean_acts = safe_load(mean_acts_path)
    sparsity = safe_load(sparsity_path)

    num_features = labels.shape[0]
    available_top_k = labels.shape[1]
    top_k = min(top_k, available_top_k)

    purities = []
    majority_labels = []
    unique_label_counts = []
    valid_top_counts = []
    top_image_indices = []
    top_activation_values = []
    top_labels = []

    for feature_id in range(num_features):
        valid = values[feature_id, :top_k] > 0

        feature_labels = labels[feature_id, :top_k][valid].tolist()
        feature_values = values[feature_id, :top_k][valid].tolist()
        feature_indices = indices[feature_id, :top_k][valid].tolist()

        valid_count = len(feature_labels)

        if valid_count == 0:
            purity = 0.0
            majority_label = -1
            unique_count = 0
            first_image_index = -1
            first_activation_value = 0.0
            first_label = -1
        else:
            counts = Counter(feature_labels)
            majority_label, majority_count = counts.most_common(1)[0]

            purity = majority_count / valid_count
            unique_count = len(counts)

            first_image_index = int(feature_indices[0])
            first_activation_value = float(feature_values[0])
            first_label = int(feature_labels[0])

        purities.append(purity)
        majority_labels.append(majority_label)
        unique_label_counts.append(unique_count)
        valid_top_counts.append(valid_count)
        top_image_indices.append(first_image_index)
        top_activation_values.append(first_activation_value)
        top_labels.append(first_label)

    purities_t = torch.tensor(purities, dtype=torch.float32)
    majority_labels_t = torch.tensor(majority_labels, dtype=torch.long)
    unique_label_counts_t = torch.tensor(unique_label_counts, dtype=torch.long)
    valid_top_counts_t = torch.tensor(valid_top_counts, dtype=torch.long)

    torch.save(purities_t, feature_dir / f"label_purity_top{top_k}.pt")
    torch.save(majority_labels_t, feature_dir / f"majority_label_top{top_k}.pt")
    torch.save(unique_label_counts_t, feature_dir / f"unique_label_count_top{top_k}.pt")
    torch.save(valid_top_counts_t, feature_dir / f"valid_top_count_top{top_k}.pt")

    # Useful ranking scores
    num_activating_estimate = sparsity * num_features

    df = pd.DataFrame(
        {
            "layer": layer_name,
            "feature_id": list(range(num_features)),
            "label_purity": purities,
            "majority_label": majority_labels,
            "unique_label_count": unique_label_counts,
            "valid_top_count": valid_top_counts,
            "mean_activation": mean_acts.tolist(),
            "sparsity": sparsity.tolist(),
            "top_image_index": top_image_indices,
            "top_activation_value": top_activation_values,
            "top_label": top_labels,
        }
    )

    df["has_saved_folder"] = df["feature_id"].apply(
        lambda feature_id: (feature_dir / str(feature_id)).is_dir()
    )

    df["class_specific_score"] = df["label_purity"] * df["mean_activation"]
    df["sparse_class_specific_score"] = (
        df["label_purity"] * df["mean_activation"] * df["sparsity"]
    )

    csv_path = feature_dir / f"feature_summary_top{top_k}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved: {feature_dir / f'label_purity_top{top_k}.pt'}")
    print(f"Saved: {feature_dir / f'majority_label_top{top_k}.pt'}")
    print(f"Saved: {feature_dir / f'unique_label_count_top{top_k}.pt'}")
    print(f"Saved: {csv_path}")

    print("\nQuick summary:")
    print(f"Number of features: {num_features}")
    print(f"Mean purity: {purities_t.mean().item():.4f}")
    print(f"Max purity: {purities_t.max().item():.4f}")
    print(f"Features with purity >= 0.8: {(purities_t >= 0.8).sum().item()}")
    print(f"Features with purity >= 0.9: {(purities_t >= 0.9).sum().item()}")

def main():
    for dataset_name, layer_dirs in DATASET_LAYER_DIRS.items():
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        for layer_name, feature_dir in layer_dirs.items():
            feature_dir = Path(feature_dir)

            if not feature_dir.exists():
                print(
                    f"Skipping missing directory: "
                    f"{dataset_name} / {layer_name} -> {feature_dir}"
                )
                continue

            compute_stats_for_layer(
                layer_name=f"{dataset_name}_{layer_name}",
                feature_dir=feature_dir,
                top_k=TOP_K_FOR_PURITY,
            )


if __name__ == "__main__":
    main()
