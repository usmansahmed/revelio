import torch
from training.config import SDSAERunnerConfig
from training.k_sparse_autoencoder import KSparseAutoencoder
from training.sd_activations_store import SDActivationsStore
from training.save_feature import save_features

checkpoint = torch.load("Checkpoints/wfsothky/final_k_sparse_autoencoder_/home/woody/rlvl/rlvl172v/revelio/SD-kSAE/caltech101/SDv1-5/timestep_25/mid_block_10_mid_block_81920.pt", map_location="cuda")
cfg = checkpoint["cfg"]

model = KSparseAutoencoder(cfg)
model.load_state_dict(checkpoint["state_dict"])
model.to(cfg.device)
model.eval()

activation_store = SDActivationsStore(cfg)

save_features(
    model,
    activation_store,
    number_of_images=24790,
    number_of_max_activating_images=20,
)
