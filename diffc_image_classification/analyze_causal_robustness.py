import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def bootstrap_ci(values, n_bootstrap=10000, seed=42):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return None, None

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(n_bootstrap, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])

    return float(low), float(high)


def compute_stats(rows, bootstrap_samples=10000, seed=42):
    if not rows:
        return {
            "n": 0,
            "mean_margin_difference": None,
            "median_margin_difference": None,
            "ci_95_low": None,
            "ci_95_high": None,
            "positive_images": 0,
            "negative_images": 0,
            "fraction_positive": None,
            "wilcoxon_p": None,
            "sign_test_p": None,
        }

    values = np.array(
        [row["margin_drop_difference"] for row in rows],
        dtype=float,
    )

    mean_diff = float(values.mean())
    median_diff = float(np.median(values))

    ci_low, ci_high = bootstrap_ci(
        values,
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )

    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    nonzero = positive + negative

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

    if nonzero > 0:
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

    return {
        "n": len(rows),
        "mean_margin_difference": mean_diff,
        "median_margin_difference": median_diff,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "positive_images": positive,
        "negative_images": negative,
        "fraction_positive": positive / len(rows),
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
    }


def benjamini_hochberg(p_values):
    """
    Return Benjamini-Hochberg FDR-adjusted p-values.
    None/NaN values remain None.
    """
    adjusted = [None] * len(p_values)

    valid = [
        (i, float(p))
        for i, p in enumerate(p_values)
        if p is not None and not np.isnan(p)
    ]

    if not valid:
        return adjusted

    valid.sort(key=lambda x: x[1])
    m = len(valid)

    raw_adjusted = []

    for rank, (original_index, p) in enumerate(valid, start=1):
        raw_adjusted.append([
            original_index,
            min(p * m / rank, 1.0),
        ])

    # Enforce monotonicity from largest p-value downward.
    running_min = 1.0

    for i in range(len(raw_adjusted) - 1, -1, -1):
        original_index, value = raw_adjusted[i]
        running_min = min(running_min, value)
        adjusted[original_index] = running_min

    return adjusted


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        required=True,
        help="Directory containing per_class/*.json",
    )

    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    per_class_dir = run_dir / "per_class"

    if not per_class_dir.exists():
        raise FileNotFoundError(
            f"Could not find: {per_class_dir}"
        )

    rows = []

    for json_path in sorted(per_class_dir.glob("class_*.json")):
        with json_path.open("r") as f:
            data = json.load(f)

        class_id = data["class_id"]
        class_name = data["class_name"]

        per_image = data.get("per_image_results", [])

        # Any image where a targeted intervention occurred.
        treated = [
            row
            for row in per_image
            if row["number_target_neurons_ablated"] > 0
        ]

        # Robustness subset:
        # intervention occurred AND Diff-C was initially correct.
        correct_treated = [
            row
            for row in treated
            if row["original_pred"] == row["true_label"]
        ]

        all_stats = compute_stats(
            treated,
            bootstrap_samples=args.bootstrap_samples,
            seed=42 + class_id,
        )

        correct_stats = compute_stats(
            correct_treated,
            bootstrap_samples=args.bootstrap_samples,
            seed=1000 + class_id,
        )

        rows.append({
            "class_id": class_id,
            "class_name": class_name,

            "n_selected_features":
                data.get("selected_feature_count", 0),

            "n_treated":
                all_stats["n"],

            "all_mean_margin_difference":
                all_stats["mean_margin_difference"],

            "all_ci_95_low":
                all_stats["ci_95_low"],

            "all_ci_95_high":
                all_stats["ci_95_high"],

            "all_fraction_positive":
                all_stats["fraction_positive"],

            "all_wilcoxon_p":
                all_stats["wilcoxon_p"],

            "n_correct_treated":
                correct_stats["n"],

            "correct_mean_margin_difference":
                correct_stats["mean_margin_difference"],

            "correct_median_margin_difference":
                correct_stats["median_margin_difference"],

            "correct_ci_95_low":
                correct_stats["ci_95_low"],

            "correct_ci_95_high":
                correct_stats["ci_95_high"],

            "correct_positive_images":
                correct_stats["positive_images"],

            "correct_negative_images":
                correct_stats["negative_images"],

            "correct_fraction_positive":
                correct_stats["fraction_positive"],

            "correct_wilcoxon_p":
                correct_stats["wilcoxon_p"],

            "correct_sign_test_p":
                correct_stats["sign_test_p"],
        })

    df = pd.DataFrame(rows).sort_values("class_id")

    # Multiple-comparison correction across classes that were actually tested.
    df["all_wilcoxon_fdr"] = benjamini_hochberg(
        df["all_wilcoxon_p"].tolist()
    )

    df["correct_wilcoxon_fdr"] = benjamini_hochberg(
        df["correct_wilcoxon_p"].tolist()
    )

    if args.output is None:
        output_path = (
            run_dir / "causal_robustness_summary.csv"
        )
    else:
        output_path = Path(args.output)

    df.to_csv(output_path, index=False)

    display_columns = [
        "class_id",
        "class_name",
        "n_treated",
        "all_mean_margin_difference",
        "all_ci_95_low",
        "all_ci_95_high",
        "all_wilcoxon_fdr",
        "n_correct_treated",
        "correct_mean_margin_difference",
        "correct_ci_95_low",
        "correct_ci_95_high",
        "correct_fraction_positive",
        "correct_wilcoxon_fdr",
    ]

    print()
    print(df[display_columns].to_string(index=False))
    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()