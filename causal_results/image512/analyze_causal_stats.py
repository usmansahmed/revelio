import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def bootstrap_ci(values, n_bootstrap=10000, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()

    lower, upper = np.percentile(means, [2.5, 97.5])

    return float(lower), float(upper)


def analyze_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    treated = [
        row for row in data["per_image_results"]
        if row["number_target_neurons_ablated"] > 0
    ]

    differences = np.array([
        row["targeted_margin_drop"] - row["random_margin_drop"]
        for row in treated
    ])

    mean_difference = differences.mean()
    median_difference = np.median(differences)

    ci_low, ci_high = bootstrap_ci(differences)

    # Paired non-parametric test:
    # H0: targeted and random margin drops are the same.
    wilcoxon_result = wilcoxon(
        differences,
        alternative="greater",
        zero_method="wilcox",
    )

    positive = int((differences > 0).sum())
    negative = int((differences < 0).sum())

    # Ignore exact zeros for the sign test.
    n_nonzero = positive + negative

    if n_nonzero > 0:
        sign_result = binomtest(
            positive,
            n=n_nonzero,
            p=0.5,
            alternative="greater",
        )
        sign_p = sign_result.pvalue
    else:
        sign_p = 1.0

    return {
        "class_id": data["config"]["target_class"],
        "class_name": data["config"]["target_class_name"],
        "n_treated": len(treated),
        "mean_margin_difference": float(mean_difference),
        "median_margin_difference": float(median_difference),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "positive_images": positive,
        "negative_images": negative,
        "fraction_positive": float(positive / len(treated)),
        "wilcoxon_p": float(wilcoxon_result.pvalue),
        "sign_test_p": float(sign_p),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "json_files",
        nargs="+",
        help="Causal-ablation JSON result files",
    )

    parser.add_argument(
        "--output",
        default="causal_statistical_summary.csv",
    )

    args = parser.parse_args()

    results = [
        analyze_file(Path(path))
        for path in args.json_files
    ]

    df = pd.DataFrame(results)

    print()
    print(df.to_string(index=False))
    print()

    df.to_csv(args.output, index=False)

    print("Saved summary to:", args.output)


if __name__ == "__main__":
    main()
