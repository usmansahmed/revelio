import argparse
import csv
from pathlib import Path

import torch


def load_tensor(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        required=True,
        help="Directory containing label_purity_top10.pt etc.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--min_purity",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--min_valid",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)

    label_purity = load_tensor(feature_dir / "label_purity_top10.pt")
    majority_label = load_tensor(feature_dir / "majority_label_top10.pt")
    valid_count = load_tensor(feature_dir / "valid_top_count_top10.pt")
    mean_acts = load_tensor(feature_dir / "sae_mean_acts.pt")
    sparsity = load_tensor(feature_dir / "sae_sparsity.pt")

    rows = []

    for class_id in range(args.num_classes):
        mask = (
            (majority_label == class_id)
            & (label_purity >= args.min_purity)
            & (valid_count >= args.min_valid)
        )

        feature_ids = torch.nonzero(mask, as_tuple=False).flatten()

        if len(feature_ids) == 0:
            rows.append({
                "class_id": class_id,
                "candidate_count": 0,
                "avg_purity": 0.0,
                "max_purity": 0.0,
                "avg_valid_count": 0.0,
                "avg_mean_activation": 0.0,
                "avg_sparsity": 0.0,
                "max_sparsity": 0.0,
                "top_feature_ids": "",
            })
            continue

        scores = label_purity[feature_ids] * mean_acts[feature_ids]
        order = torch.argsort(scores, descending=True)
        top_features = feature_ids[order[:10]].tolist()

        rows.append({
            "class_id": class_id,
            "candidate_count": int(len(feature_ids)),
            "avg_purity": float(label_purity[feature_ids].mean().item()),
            "max_purity": float(label_purity[feature_ids].max().item()),
            "avg_valid_count": float(valid_count[feature_ids].float().mean().item()),
            "avg_mean_activation": float(mean_acts[feature_ids].mean().item()),
            "avg_sparsity": float(sparsity[feature_ids].mean().item()),
            "max_sparsity": float(sparsity[feature_ids].max().item()),
            "top_feature_ids": " ".join(str(x) for x in top_features),
        })

    rows = sorted(
        rows,
        key=lambda row: (
            row["candidate_count"],
            row["avg_sparsity"],
            row["avg_mean_activation"],
        ),
        reverse=True,
    )

    print()
    print("Feature directory:", feature_dir)
    print("Minimum purity:", args.min_purity)
    print("Minimum valid top images:", args.min_valid)
    print()
    print("Top classes by candidate feature count:")
    print(
        "rank,class_id,candidate_count,avg_purity,"
        "avg_sparsity,max_sparsity,avg_mean_activation,top_feature_ids"
    )

    for rank, row in enumerate(rows[:args.top_n], start=1):
        print(
            f"{rank},"
            f"{row['class_id']},"
            f"{row['candidate_count']},"
            f"{row['avg_purity']:.4f},"
            f"{row['avg_sparsity']:.6f},"
            f"{row['max_sparsity']:.6f},"
            f"{row['avg_mean_activation']:.6f},"
            f"{row['top_feature_ids']}"
        )

    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print()
        print("Saved CSV to:", output_csv)


if __name__ == "__main__":
    main()
