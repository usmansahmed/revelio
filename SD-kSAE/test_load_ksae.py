import torch


ckpt_path = (
    "/home/woody/rlvl/rlvl172v/revelio/SD-kSAE/Checkpoints/d0p5pu4r/"
    "final_k_sparse_autoencoder_/home/woody/rlvl/rlvl172v/revelio/SD-kSAE/"
    "oxfordpet/SDv1-5/timestep_25/up_blocks_1_10_up_blocks_1_81920.pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt["state_dict"]
cfg = ckpt["cfg"]

W_enc = state["W_enc"].to(device)
b_enc = state["b_enc"].to(device)
W_dec = state["W_dec"].to(device)
b_dec = state["b_dec"].to(device)

d_in = cfg.d_in
k = cfg.k
n_features = W_enc.shape[1]

print("Device:", device)
print("d_in:", d_in)
print("k:", k)
print("n_features:", n_features)
print("W_enc:", W_enc.shape)
print("W_dec:", W_dec.shape)

# Dummy pooled up_ft1 feature: [batch, d_in]
x = torch.randn(4, d_in, device=device)

# Encode
pre_acts = (x - b_dec) @ W_enc + b_enc
acts = torch.relu(pre_acts)

top_values, top_indices = torch.topk(acts, k=k, dim=-1)

sparse_acts = torch.zeros_like(acts)
sparse_acts.scatter_(dim=-1, index=top_indices, src=top_values)

# Decode
x_recon = sparse_acts @ W_dec + b_dec

print("Input shape:", x.shape)
print("Sparse acts shape:", sparse_acts.shape)
print("Reconstruction shape:", x_recon.shape)
print("Nonzero SAE features per sample:", (sparse_acts > 0).sum(dim=1))
