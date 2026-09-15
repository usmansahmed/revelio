import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from torchvision.utils import make_grid, save_image


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=4)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tensor(path, device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing tensor file: {path}")
    return torch.load(path, map_location=device)


def load_feature_stats(feature_dir, device):
    feature_dir = Path(feature_dir)
    return {
        "label_purity": load_tensor(feature_dir / "label_purity_top10.pt", device).float(),
        "majority_label": load_tensor(feature_dir / "majority_label_top10.pt", device).long(),
        "valid_count": load_tensor(feature_dir / "valid_top_count_top10.pt", device).long(),
        "mean_acts": load_tensor(feature_dir / "sae_mean_acts.pt", device).float(),
        "sparsity": load_tensor(feature_dir / "sae_sparsity.pt", device).float(),
    }


def load_ksae(checkpoint_path, device, default_k=32):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["state_dict"]
    if "cfg" in checkpoint and hasattr(checkpoint["cfg"], "k"):
        k = checkpoint["cfg"].k
    else:
        k = default_k

    return {
        "W_enc": state["W_enc"].to(device).float(),
        "b_enc": state["b_enc"].to(device).float(),
        "W_dec": state["W_dec"].to(device).float(),
        "b_dec": state["b_dec"].to(device).float(),
        "k": k,
        "n_features": state["W_enc"].shape[1],
    }


def ksae_encode(x, ksae):
    pre_acts = (x - ksae["b_dec"]) @ ksae["W_enc"] + ksae["b_enc"]
    top_values, top_indices = torch.topk(pre_acts, k=ksae["k"], dim=-1)
    sparse_acts = torch.zeros_like(pre_acts)
    sparse_acts.scatter_(dim=-1, index=top_indices, src=top_values)
    return sparse_acts


def ksae_decode(sparse_acts, ksae):
    return sparse_acts @ ksae["W_dec"] + ksae["b_dec"]


def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of these columns: {candidates}")


def get_target_feature_ids_from_csv(feature_summary_path, target_class, min_purity, min_valid):
    df = pd.read_csv(feature_summary_path)

    feature_col = pick_column(df, ["feature_id", "neuron_id", "fid"])
    majority_col = pick_column(df, ["majority_label", "majority_class", "top_label"])
    purity_col = pick_column(df, ["label_purity", "purity"])
    valid_col = pick_column(df, ["valid_count", "valid_top_count", "top_count"])

    selected = df[
        (df[majority_col] == target_class)
        & (df[purity_col] >= min_purity)
        & (df[valid_col] >= min_valid)
    ].copy()

    feature_ids = selected[feature_col].astype(int).tolist()
    feature_ids.sort()
    return feature_ids


def score_active_candidates(active_ids, sparse_row, stats, ranking_method):
    purity = stats["label_purity"][active_ids]
    mean_acts = stats["mean_acts"][active_ids]
    sparsity = stats["sparsity"][active_ids]
    actual_activation = sparse_row[active_ids]

    if ranking_method == "actual_activation":
        return actual_activation
    if ranking_method == "purity_actual_activation":
        return purity * actual_activation
    if ranking_method == "purity_mean_activation":
        return purity * mean_acts
    if ranking_method == "sparse_class_specific":
        return purity * mean_acts * sparsity
    if ranking_method == "purity":
        return purity
    if ranking_method == "sparsity":
        return sparsity

    raise ValueError(f"Unknown ranking method: {ranking_method}")


def select_active_target_ids(sparse_row, allowed_ids, stats, max_modify_per_image, ranking_method):
    active_ids = torch.nonzero(sparse_row != 0, as_tuple=False).flatten()
    if active_ids.numel() == 0:
        return active_ids[:0]

    allowed_ids = allowed_ids.to(active_ids.device)
    target_ids = active_ids[torch.isin(active_ids, allowed_ids)]

    if target_ids.numel() == 0:
        return target_ids

    scores = score_active_candidates(target_ids, sparse_row, stats, ranking_method)
    order = torch.argsort(scores, descending=True)
    target_ids = target_ids[order[:max_modify_per_image]]
    return target_ids


def rank_global_target_ids(allowed_ids, stats):
    if allowed_ids.numel() == 0:
        return allowed_ids
    scores = stats["label_purity"][allowed_ids] * stats["mean_acts"][allowed_ids]
    order = torch.argsort(scores, descending=True)
    return allowed_ids[order]


def resolve_hook_module(unet, diffusion_layer):
    if diffusion_layer.startswith("up_ft:"):
        idx = int(diffusion_layer.split(":")[1])
        return unet.up_blocks[idx]
    if diffusion_layer == "mid":
        return unet.mid_block
    raise ValueError(
        f"Unsupported diffusion_layer '{diffusion_layer}'. "
        f"For now this script supports 'up_ft:x' and 'mid'."
    )


class PooledSAEIntervention:
    def __init__(self, mode, ksae, stats, target_feature_ids, cfg):
        self.mode = mode
        self.ksae = ksae
        self.stats = stats
        self.cfg = cfg
        self.enabled = True
        self.selected_ids = []
        self.sum_selected_activation = 0.0

        self.allowed_ids = torch.tensor(
            target_feature_ids,
            device=ksae["W_enc"].device,
            dtype=torch.long,
        )
        self.global_ranked_ids = rank_global_target_ids(self.allowed_ids, stats)

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return output

        x = output[0] if isinstance(output, tuple) else output
        x_dtype = x.dtype
        pooled = x.mean(dim=(2, 3)).float()
        sparse = ksae_encode(pooled, self.ksae)
        sparse_mod = sparse.clone()

        sparse_row = sparse[0]
        selected = sparse_row[:0]

        if self.mode == "ablation":
            selected = select_active_target_ids(
                sparse_row=sparse_row,
                allowed_ids=self.allowed_ids,
                stats=self.stats,
                max_modify_per_image=self.cfg["max_modify_per_image"],
                ranking_method=self.cfg["active_feature_ranking"],
            )
            if selected.numel() > 0:
                sparse_mod[0, selected] = 0.0

        elif self.mode == "amplification":
            selected = select_active_target_ids(
                sparse_row=sparse_row,
                allowed_ids=self.allowed_ids,
                stats=self.stats,
                max_modify_per_image=self.cfg["max_modify_per_image"],
                ranking_method=self.cfg["active_feature_ranking"],
            )
            if selected.numel() > 0:
                sparse_mod[0, selected] = (
                    sparse_mod[0, selected] * self.cfg["amplification_factor"]
                )

        elif self.mode == "insertion":
            max_insert = self.cfg["max_modify_per_image"]
            selected = self.global_ranked_ids[:max_insert]
            if selected.numel() > 0:
                insert_values = (
                    self.stats["mean_acts"][selected] * self.cfg["insertion_value_scale"]
                )
                sparse_mod[0, selected] = torch.maximum(
                    sparse_mod[0, selected],
                    insert_values,
                )

        elif self.mode == "baseline":
            selected = sparse_row[:0]

        else:
            raise ValueError(f"Unknown intervention mode: {self.mode}")

        recon_pooled = ksae_decode(sparse, self.ksae)
        mod_pooled = ksae_decode(sparse_mod, self.ksae)
        delta = (mod_pooled - recon_pooled)[:, :, None, None].to(x_dtype)
        y = x + delta

        self.selected_ids = [int(fid) for fid in selected.detach().cpu().tolist()]
        self.sum_selected_activation = float(
            sparse_row[selected].sum().detach().cpu().item()
        ) if selected.numel() > 0 else 0.0

        if isinstance(output, tuple):
            out = list(output)
            out[0] = y
            return tuple(out)
        return y


def decode_latents(vae, latents):
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        decoded = vae.decode(latents).sample
    return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)


def make_noisy_latents(sd_model, images, timestep, seed):
    with torch.no_grad():
        latents = sd_model.vae.encode(images).latent_dist.mode()
        latents = latents * sd_model.vae.config.scaling_factor

    generator = torch.Generator(device=images.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=images.device,
        dtype=latents.dtype,
    )

    t = torch.tensor([timestep], dtype=torch.long, device=images.device)
    latents_noisy = sd_model.scheduler.add_noise(latents, noise, t)
    return latents, noise, latents_noisy


def predict_x0(sd_model, latents_noisy, prompt_embeds, timestep):
    t = torch.tensor([timestep], dtype=torch.long, device=latents_noisy.device)

    with torch.no_grad():
        unet_out = sd_model.unet(
            latents_noisy,
            t,
            up_ft_indices=[len(sd_model.unet.up_blocks) - 1],
            encoder_hidden_states=prompt_embeds,
        )

    noise_pred = unet_out["sample"]

    alphas_cumprod = sd_model.scheduler.alphas_cumprod.to(
        device=latents_noisy.device,
        dtype=latents_noisy.dtype,
    )

    alpha_t = alphas_cumprod[timestep]
    sqrt_alpha_t = alpha_t.sqrt().view(1, 1, 1, 1)
    sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt().view(1, 1, 1, 1)

    x0 = (
        latents_noisy - sqrt_one_minus_alpha_t * noise_pred
    ) / sqrt_alpha_t

    return x0


def prepare_source_indices(hf_dataset, cfg):
    if cfg.get("source_indices"):
        return cfg["source_indices"]

    if cfg.get("source_class", None) is None:
        raise ValueError("Provide either source_indices or source_class in config.")

    labels_all = hf_dataset["label"]
    source_class = cfg["source_class"]
    selected = [i for i, y in enumerate(labels_all) if int(y) == source_class]
    return selected[:cfg["max_source_images"]]


def tensor_to_png(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    set_seed(cfg.get("random_seed", 42))

    sys.path.insert(0, cfg["diffc_dir"])
    sys.path.insert(1, cfg["sd_ksae_dir"])

    from helpers.dataset import HuggingFaceImageDataset, load_huggingface_dataset
    from constants import model_base_dict, diffusion_transformers_val, clip_transforms
    from models import ImageClassifer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading feature stats and k-SAE...")
    stats = load_feature_stats(cfg["feature_dir"], device)
    ksae = load_ksae(
        cfg["ksae_checkpoint_path"],
        device=device,
        default_k=cfg.get("ksae_k", 32),
    )

    target_feature_ids = get_target_feature_ids_from_csv(
        feature_summary_path=cfg["feature_summary_path"],
        target_class=cfg["target_class"],
        min_purity=cfg["min_purity"],
        min_valid=cfg["min_valid"],
    )

    print(f"Target class: {cfg['target_class']}")
    print(f"Selected high-purity target features: {len(target_feature_ids)}")

    print("Loading dataset...")
    hf_dataset = load_huggingface_dataset(cfg["dataset_flag"], split=cfg.get("split", "test"))
    class_names = hf_dataset.features["label"].names
    target_class_name = class_names[cfg["target_class"]]
    source_class_name = class_names[cfg["source_class"]] if cfg.get("source_class", None) is not None else "custom_indices"

    test_dataset = HuggingFaceImageDataset(
        hf_dataset,
        diffusion_transformers_val,
        clip_transforms,
    )

    source_indices = prepare_source_indices(hf_dataset, cfg)
    print("Source class:", cfg.get("source_class", None), source_class_name)
    print("Number of source images:", len(source_indices))
    print("First source indices:", source_indices[:10])

    diffc_config = {
        "dataset_flag": cfg["dataset_flag"],
        "output_dir": "",
        "seed": cfg.get("random_seed", 42),
        "model_name": cfg["model_name"],
        "diffusion_timestep": cfg["diffusion_timestep"],
        "diffusion_layer": cfg["diffusion_layer"],
        "learning_rate": cfg.get("learning_rate", 1e-4),
        "num_epochs": cfg.get("num_epochs", 90),
        "batch_size": 1,
        "prompt_type": cfg.get("prompt_type", "empty"),
        "pooling_strategy": cfg.get("pooling_strategy", "GAP"),
        "dropout_rate": cfg["dropout_rate"],
        "num_classes": cfg["num_classes"],
        "num_devices": 1,
        "feature_model": model_base_dict[cfg["model_name"]],
        "diffusion_step_type": cfg.get("diffusion_step_type", "onestep"),
        "device": device,
        "input_channels": cfg["input_channels"],
    }

    print("Loading Stable Diffusion feature extractor via ImageClassifer...")
    model = ImageClassifer(diffc_config).to(device)
    model.eval()

    sd_model = model.feature_model
    from diffusers import AutoencoderKL
    vae_decoder = AutoencoderKL.from_pretrained(
        cfg["model_name"],
        subfolder="vae",
        torch_dtype=sd_model.dtype
    ).to(device)

    vae_decoder.eval()
    hook_module = resolve_hook_module(sd_model.unet, cfg["diffusion_layer"])

    out_dir = Path(cfg["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "target_class": cfg["target_class"],
        "target_class_name": target_class_name,
        "source_class": cfg.get("source_class", None),
        "source_class_name": source_class_name,
        "n_selected_features": len(target_feature_ids),
        "source_indices": source_indices,
        "images": [],
    }

    for source_index in source_indices:
        diffusion_image, _, label, _ = test_dataset[source_index]
        images = diffusion_image.unsqueeze(0).to(device)

        image_seed = cfg.get("random_seed", 42) + int(source_index)
        _, noise, latents_noisy = make_noisy_latents(
            sd_model,
            images,
            cfg["diffusion_timestep"],
            seed=image_seed,
        )

        prompt_embeds = sd_model.empty_prompt_embeds.repeat(1, 1, 1).to(
            device=device,
            dtype=sd_model.dtype,
        )

        item_dir = out_dir / f"img_{source_index:05d}"
        item_dir.mkdir(parents=True, exist_ok=True)

        source_png = ((images[0].detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
        tensor_to_png(source_png, item_dir / "source.png")

        conditions = ["baseline", "ablation", "amplification", "insertion"]
        saved_tensors = [source_png]
        image_result = {
            "source_index": int(source_index),
            "source_label": int(label),
            "source_label_name": class_names[int(label)],
            "conditions": {},
        }

        for mode in conditions:
            if mode == "baseline":
                x0 = predict_x0(
                    sd_model=sd_model,
                    latents_noisy=latents_noisy,
                    prompt_embeds=prompt_embeds,
                    timestep=cfg["diffusion_timestep"],
                )
                selected_ids = []
                sum_selected_activation = 0.0
            else:
                intervention = PooledSAEIntervention(
                    mode=mode,
                    ksae=ksae,
                    stats=stats,
                    target_feature_ids=target_feature_ids,
                    cfg=cfg,
                )
                hook_handle = hook_module.register_forward_hook(intervention)

                try:
                    x0 = predict_x0(
                        sd_model=sd_model,
                        latents_noisy=latents_noisy,
                        prompt_embeds=prompt_embeds,
                        timestep=cfg["diffusion_timestep"],
                    )
                finally:
                    hook_handle.remove()

                selected_ids = intervention.selected_ids
                sum_selected_activation = intervention.sum_selected_activation

            recon_img = decode_latents(vae_decoder, x0)[0].detach().cpu()
            tensor_to_png(recon_img, item_dir / f"{mode}.png")
            saved_tensors.append(recon_img)

            image_result["conditions"][mode] = {
                "selected_feature_ids": selected_ids,
                "n_selected": len(selected_ids),
                "sum_selected_activation": sum_selected_activation,
            }

        grid = make_grid(saved_tensors, nrow=len(saved_tensors))
        tensor_to_png(grid, item_dir / "grid.png")
        summary["images"].append(image_result)

        print(
            f"Done source_index={source_index} "
            f"label={class_names[int(label)]} "
            f"saved_to={item_dir}"
        )

    save_json(summary, out_dir / "summary.json")
    print("Saved summary to:", out_dir / "summary.json")


if __name__ == "__main__":
    main()