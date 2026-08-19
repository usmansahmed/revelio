import torch
from pathlib import Path


feature_dir = Path(
    "/home/woody/rlvl/rlvl172v/revelio/SD-kSAE/features/oxfordpet/SDv1-5/step25_upft1"
)

target_class = 19
min_purity = 0.6
min_valid = 5
top_n = 20

label_purity = torch.load(feature_dir / "label_purity_top10.pt", map_location="cpu")
majority_label = torch.load(feature_dir / "majority_label_top10.pt", map_location="cpu")
valid_count = torch.load(feature_dir / "valid_top_count_top10.pt", map_location="cpu")
mean_acts = torch.load(feature_dir / "sae_mean_acts.pt", map_location="cpu")
sparsity = torch.load(feature_dir / "sae_sparsity.pt", map_location="cpu")

mask = (
    (majority_label == target_class)
    & (label_purity >= min_purity)
    & (valid_count >= min_valid)
)

feature_ids = torch.nonzero(mask, as_tuple=False).flatten()

print("Target class:", target_class)
print("Matching features:", len(feature_ids))

if len(feature_ids) == 0:
    raise SystemExit("No matching features found.")

# Simple ranking score. We can change this later.
scores = label_purity[feature_ids] * mean_acts[feature_ids]

order = torch.argsort(scores, descending=True)
selected = feature_ids[order[:top_n]]

print()
print("Top candidate features:")
print("feature_id,purity,valid_count,mean_act,sparsity,score")

for fid in selected.tolist():
    print(
        f"{fid},"
        f"{label_purity[fid].item():.4f},"
        f"{valid_count[fid].item()},"
        f"{mean_acts[fid].item():.6f},"
        f"{sparsity[fid].item():.6f},"
        f"{(label_purity[fid] * mean_acts[fid]).item():.6f}"
    )
