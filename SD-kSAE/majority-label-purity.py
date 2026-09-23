import pandas as pd

files = {
    "bottleneck": "bottleneck/feature_summary_top10.csv",
    "up_ft0": "upft0/feature_summary_top10.csv",
    "up_ft1": "upft1/feature_summary_top10.csv",
    "up_ft2": "upft2/feature_summary_top10.csv",
}

number_of_images = 3334

for layer, path in files.items():
    df = pd.read_csv(path)

    df["num_activating"] = df["sparsity"] * number_of_images
    df["ranking_score"] = (
        df["mean_activation"]
        * (df["num_activating"] > 10)
    )

    top1000 = (
        df.sort_values("ranking_score", ascending=False)
          .head(1000)
    )

    purity = top1000["label_purity"].mean()

    print(f"{layer}: {purity:.6f} ({purity * 100:.2f}%)")