import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_mean(values):
    return float(np.mean(values)) if values else None


def safe_median(values):
    return float(np.median(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Sufficiency run directory containing per_target_class/")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    detail_dir = run_dir / "per_target_class"

    if not detail_dir.exists():
        raise FileNotFoundError(f"Could not find: {detail_dir}")

    summary_rows = []
    per_image_rows = []

    for path in sorted(detail_dir.glob("target_*.json")):
        with path.open("r") as f:
            data = json.load(f)

        target_class = data["target_class"]
        target_name = data["target_class_name"]
        rows = data.get("per_image_results", [])

        if not rows:
            summary_rows.append({
                "target_class": target_class,
                "target_class_name": target_name,
                "n_source_images": 0,
            })
            continue

        original = [r["original_target_margin"] for r in rows]
        targeted = [r["targeted_target_margin"] for r in rows]
        control = [r["control_target_margin"] for r in rows]

        targeted_inc = [t - o for t, o in zip(targeted, original)]
        control_inc = [c - o for c, o in zip(control, original)]
        targeted_vs_control = [t - c for t, c in zip(targeted, control)]

        targeted_cross_zero = sum(o <= 0 and t > 0 for o, t in zip(original, targeted))
        control_cross_zero = sum(o <= 0 and c > 0 for o, c in zip(original, control))

        summary_rows.append({
            "target_class": target_class,
            "target_class_name": target_name,
            "n_source_images": len(rows),

            "mean_original_target_margin": safe_mean(original),
            "mean_targeted_target_margin": safe_mean(targeted),
            "mean_control_target_margin": safe_mean(control),

            "median_original_target_margin": safe_median(original),
            "median_targeted_target_margin": safe_median(targeted),
            "median_control_target_margin": safe_median(control),

            "mean_targeted_margin_increase": safe_mean(targeted_inc),
            "mean_control_margin_increase": safe_mean(control_inc),
            "mean_targeted_minus_control_margin": safe_mean(targeted_vs_control),

            "original_target_top1_count": sum(x > 0 for x in original),
            "targeted_target_top1_count": sum(x > 0 for x in targeted),
            "control_target_top1_count": sum(x > 0 for x in control),

            "targeted_crossed_zero_count": targeted_cross_zero,
            "control_crossed_zero_count": control_cross_zero,
            "targeted_crossed_zero_fraction": targeted_cross_zero / len(rows),
            "control_crossed_zero_fraction": control_cross_zero / len(rows),

            "fraction_targeted_margin_greater_than_original": float(np.mean(np.array(targeted) > np.array(original))),
            "fraction_targeted_margin_greater_than_control": float(np.mean(np.array(targeted) > np.array(control))),
        })

        for r in rows:
            per_image_rows.append({
                "target_class": target_class,
                "target_class_name": target_name,
                "image_index": r["image_index"],
                "source_true_label": r["true_label"],
                "original_pred": r["original_pred"],
                "targeted_pred": r["targeted_pred"],
                "control_pred": r["control_pred"],
                "original_target_margin": r["original_target_margin"],
                "targeted_target_margin": r["targeted_target_margin"],
                "control_target_margin": r["control_target_margin"],
                "targeted_margin_increase": r["targeted_target_margin"] - r["original_target_margin"],
                "control_margin_increase": r["control_target_margin"] - r["original_target_margin"],
                "targeted_minus_control_margin": r["targeted_target_margin"] - r["control_target_margin"],
            })

    summary_df = pd.DataFrame(summary_rows).sort_values("target_class")
    per_image_df = pd.DataFrame(per_image_rows)

    output_path = Path(args.output) if args.output else run_dir / "absolute_target_margin_summary.csv"
    per_image_path = run_dir / "absolute_target_margin_per_image.csv"

    summary_df.to_csv(output_path, index=False)
    per_image_df.to_csv(per_image_path, index=False)

    display_cols = [
        "target_class", "target_class_name", "n_source_images",
        "mean_original_target_margin", "mean_targeted_target_margin", "mean_control_target_margin",
        "mean_targeted_margin_increase", "mean_control_margin_increase",
        "mean_targeted_minus_control_margin",
        "targeted_crossed_zero_count", "control_crossed_zero_count",
    ]

    print()
    print(summary_df[display_cols].to_string(index=False))
    print()
    print("Saved summary:", output_path)
    print("Saved per-image results:", per_image_path)


if __name__ == "__main__":
    main()
