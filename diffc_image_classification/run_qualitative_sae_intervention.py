import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from torchvision.utils import make_grid, save_image


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
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
    k = checkpoint["cfg"].k if "cfg" in checkpoint and hasattr(checkpoint["cfg"], "k") else default_k

    return {
        "W_enc": state["W_enc"].to(device).float(),
        "b_enc": state["b_enc"].to(device).float(),
        "W_dec": state["W_dec"].to(device).float(),
        "b_dec": state["b_dec"].to(device).float(),
        "k": int(k),
        "n_features": int(state["W_enc"].shape[1]),
        "d_in": int(state["W_enc"].shape[0]),
    }


def ksae_encode(x, ksae):
    # Exact k-SAE form used in the causal experiments: TopK directly, no ReLU.
    pre_acts = (x - ksae["b_dec"]) @ ksae["W_enc"] + ksae["b_enc"]
    top_values, top_indices = torch.topk(pre_acts, k=ksae["k"], dim=-1)
    sparse_acts = torch.zeros_like(pre_acts)
    sparse_acts.scatter_(dim=-1, index=top_indices, src=top_values)
    return sparse_acts


def ksae_decode(sparse_acts, ksae):
    return sparse_acts @ ksae["W_dec"] + ksae["b_dec"]


def select_high_purity_features(summary_df, target_class, min_purity, min_valid):
    selected = summary_df[
        (summary_df["majority_label"] == target_class)
        & (summary_df["label_purity"] >= min_purity)
        & (summary_df["valid_top_count"] >= min_valid)
    ]
    return selected["feature_id"].astype(int).tolist()


def load_reference_cache(path, device):
    cache = torch.load(path, map_location="cpu")
    if "median" not in cache or "count" not in cache:
        raise KeyError("Reference cache must contain 'median' and 'count'.")
    return cache["median"].to(device).float(), cache["count"].to(device).long()


def get_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def encode_empty_prompt(pipe, device, dtype):
    tokens = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(tokens.input_ids.to(device))[0]
    return prompt_embeds.to(device=device, dtype=dtype)


def resolve_hook_module(unet, diffusion_layer):
    if diffusion_layer.startswith("up_ft:"):
        idx = int(diffusion_layer.split(":")[1])
        if idx < 0 or idx >= len(unet.up_blocks):
            raise IndexError(f"{diffusion_layer} is invalid; UNet has {len(unet.up_blocks)} up blocks.")
        return unet.up_blocks[idx]

    if diffusion_layer == "mid":
        return unet.mid_block

    raise ValueError("This script currently supports diffusion_layer='up_ft:x' or 'mid'.")


def unpack_hook_output(output):
    if torch.is_tensor(output):
        return output, lambda new_x: new_x

    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Unsupported tuple output from hooked module.")
        return output[0], lambda new_x: (new_x,) + output[1:]

    if isinstance(output, list):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Unsupported list output from hooked module.")
        return output[0], lambda new_x: [new_x] + output[1:]

    raise TypeError(f"Unsupported hook output type: {type(output)}")


def score_active_target_features(feature_ids, sparse_row, stats, method):
    if feature_ids.numel() == 0:
        return feature_ids

    purity = stats["label_purity"][feature_ids]
    actual = sparse_row[feature_ids].abs()
    mean_acts = stats["mean_acts"][feature_ids].abs()
    sparsity = stats["sparsity"][feature_ids]

    if method == "purity_actual_activation":
        score = purity * actual
    elif method == "actual_activation":
        score = actual
    elif method == "purity_mean_activation":
        score = purity * mean_acts
    elif method == "sparse_class_specific":
        score = purity * mean_acts * sparsity
    elif method == "purity":
        score = purity
    else:
        raise ValueError(f"Unknown active_feature_ranking: {method}")

    return feature_ids[torch.argsort(score, descending=True)]


def select_active_target_ids(sparse_row, target_feature_ids, stats, max_modify, ranking_method):
    active_ids = torch.nonzero(sparse_row != 0, as_tuple=False).flatten()
    if active_ids.numel() == 0:
        return active_ids[:0]

    target_tensor = torch.tensor(target_feature_ids, dtype=torch.long, device=sparse_row.device)
    target_active = active_ids[torch.isin(active_ids, target_tensor)]
    ranked = score_active_target_features(target_active, sparse_row, stats, ranking_method)
    return ranked[:max_modify]


def match_active_controls(target_ids, sparse_row, stats, target_class):
    # Same principle as the frozen causal experiment:
    # match active unrelated neurons by absolute activation magnitude, without replacement.
    if target_ids.numel() == 0:
        return target_ids

    active_ids = torch.nonzero(sparse_row != 0, as_tuple=False).flatten()
    pool = active_ids[stats["majority_label"][active_ids] != target_class]
    matched = []

    for target_id in target_ids:
        if pool.numel() == 0:
            break
        distances = torch.abs(sparse_row[pool].abs() - sparse_row[target_id].abs())
        pos = torch.argmin(distances)
        matched.append(pool[pos])
        pool = torch.cat((pool[:pos], pool[pos + 1:]))

    return torch.stack(matched) if matched else active_ids[:0]


def rank_inactive_target_features(target_feature_ids, active_ids, reference_median, reference_count, stats, cfg):
    ids = torch.tensor(target_feature_ids, dtype=torch.long, device=reference_median.device)
    if ids.numel() == 0:
        return ids

    eligible = (
        (reference_count[ids] >= cfg["min_reference_count"])
        & (~torch.isnan(reference_median[ids]))
        & (~torch.isin(ids, active_ids))
    )
    ids = ids[eligible]
    if ids.numel() == 0:
        return ids

    ref = reference_median[ids].abs()
    purity = stats["label_purity"][ids]

    method = cfg["insertion_feature_ranking"]
    if method == "purity_reference_activation":
        score = purity * ref
    elif method == "reference_activation":
        score = ref
    elif method == "purity":
        score = purity
    else:
        raise ValueError(f"Unknown insertion_feature_ranking: {method}")

    return ids[torch.argsort(score, descending=True)]


def build_control_pool(summary_df, reference_median, reference_count, cfg, device):
    selected = summary_df[
        (summary_df["label_purity"] >= cfg["control_min_purity"])
        & (summary_df["valid_top_count"] >= cfg["control_min_valid"])
    ]
    ids = torch.tensor(selected["feature_id"].astype(int).tolist(), dtype=torch.long, device=device)
    if ids.numel() == 0:
        return ids

    eligible = (reference_count[ids] >= cfg["min_reference_count"]) & (~torch.isnan(reference_median[ids]))
    return ids[eligible]


def match_inactive_insertion_controls(
    target_ids, active_ids, target_class, control_pool_ids,
    reference_median, decoder_norms, stats
):
    # Match on reference activation magnitude, decoder-row norm and lightly on purity.
    if target_ids.numel() == 0:
        return target_ids

    active_mask = torch.zeros(reference_median.numel(), dtype=torch.bool, device=reference_median.device)
    active_mask[active_ids] = True

    pool = control_pool_ids[~active_mask[control_pool_ids]]
    pool = pool[stats["majority_label"][pool] != target_class]
    matched = []
    eps = 1e-8

    for target_id in target_ids:
        if pool.numel() == 0:
            break

        target_ref = reference_median[target_id].abs().clamp_min(eps)
        pool_ref = reference_median[pool].abs().clamp_min(eps)
        target_norm = decoder_norms[target_id].clamp_min(eps)
        pool_norm = decoder_norms[pool].clamp_min(eps)

        activation_distance = torch.abs(torch.log(pool_ref / target_ref))
        decoder_distance = torch.abs(torch.log(pool_norm / target_norm))
        purity_distance = torch.abs(stats["label_purity"][pool] - stats["label_purity"][target_id])
        distance = activation_distance + decoder_distance + 0.25 * purity_distance

        pos = torch.argmin(distance)
        matched.append(pool[pos])
        pool = torch.cat((pool[:pos], pool[pos + 1:]))

    return torch.stack(matched) if matched else control_pool_ids[:0]


def choose_source_removals(sparse_row, active_ids, target_class, stats, n_remove):
    # Preserve k by removing the weakest active non-target features.
    if n_remove <= 0:
        return active_ids[:0]

    pool = active_ids[stats["majority_label"][active_ids] != target_class]
    if pool.numel() == 0:
        return pool

    order = torch.argsort(sparse_row[pool].abs(), descending=False)
    return pool[order[:min(n_remove, pool.numel())]]


class SAEInterventionHook:
    def __init__(
        self, mode, ksae, stats, target_feature_ids, target_class,
        reference_median, reference_count, control_pool_ids, decoder_norms, cfg
    ):
        self.mode = mode
        self.ksae = ksae
        self.stats = stats
        self.target_feature_ids = target_feature_ids
        self.target_class = target_class
        self.reference_median = reference_median
        self.reference_count = reference_count
        self.control_pool_ids = control_pool_ids
        self.decoder_norms = decoder_norms
        self.cfg = cfg
        self.info = {}

    def __call__(self, module, inputs, output):
        x, repack = unpack_hook_output(output)

        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] at hook, got {tuple(x.shape)}")
        if x.shape[0] != 1:
            raise ValueError("This qualitative script expects batch size 1.")
        if x.shape[1] != self.ksae["d_in"]:
            raise ValueError(
                f"Hooked feature has C={x.shape[1]}, but k-SAE expects d_in={self.ksae['d_in']}. "
                "This usually means the wrong UNet block was hooked."
            )

        pooled = x.float().mean(dim=(2, 3))
        sparse = ksae_encode(pooled, self.ksae)
        sparse_mod = sparse.clone()
        sparse_row = sparse[0]
        active_ids = torch.nonzero(sparse_row != 0, as_tuple=False).flatten()

        info = {
            "mode": self.mode,
            "original_active_count": int(active_ids.numel()),
            "target_feature_ids": [],
            "control_feature_ids": [],
            "removed_source_feature_ids": [],
            "target_inserted_values": [],
            "control_inserted_values": [],
        }

        if self.mode == "target_ablation":
            target_ids = select_active_target_ids(
                sparse_row, self.target_feature_ids, self.stats,
                self.cfg["max_active_modify_per_image"], self.cfg["active_feature_ranking"]
            )
            sparse_mod[0, target_ids] = 0.0
            info["target_feature_ids"] = [int(x) for x in target_ids.detach().cpu().tolist()]

        elif self.mode == "control_ablation":
            target_ids = select_active_target_ids(
                sparse_row, self.target_feature_ids, self.stats,
                self.cfg["max_active_modify_per_image"], self.cfg["active_feature_ranking"]
            )
            control_ids = match_active_controls(target_ids, sparse_row, self.stats, self.target_class)
            n = min(target_ids.numel(), control_ids.numel())
            target_ids, control_ids = target_ids[:n], control_ids[:n]
            sparse_mod[0, control_ids] = 0.0
            info["target_feature_ids"] = [int(x) for x in target_ids.detach().cpu().tolist()]
            info["control_feature_ids"] = [int(x) for x in control_ids.detach().cpu().tolist()]

        elif self.mode == "target_amplification":
            target_ids = select_active_target_ids(
                sparse_row, self.target_feature_ids, self.stats,
                self.cfg["max_active_modify_per_image"], self.cfg["active_feature_ranking"]
            )
            sparse_mod[0, target_ids] *= self.cfg["amplification_factor"]
            info["target_feature_ids"] = [int(x) for x in target_ids.detach().cpu().tolist()]

        elif self.mode == "control_amplification":
            target_ids = select_active_target_ids(
                sparse_row, self.target_feature_ids, self.stats,
                self.cfg["max_active_modify_per_image"], self.cfg["active_feature_ranking"]
            )
            control_ids = match_active_controls(target_ids, sparse_row, self.stats, self.target_class)
            n = min(target_ids.numel(), control_ids.numel())
            target_ids, control_ids = target_ids[:n], control_ids[:n]
            sparse_mod[0, control_ids] *= self.cfg["amplification_factor"]
            info["target_feature_ids"] = [int(x) for x in target_ids.detach().cpu().tolist()]
            info["control_feature_ids"] = [int(x) for x in control_ids.detach().cpu().tolist()]

        elif self.mode in ("target_insertion", "control_insertion"):
            ranked_target_ids = rank_inactive_target_features(
                self.target_feature_ids, active_ids, self.reference_median,
                self.reference_count, self.stats, self.cfg
            )
            target_ids = ranked_target_ids[:self.cfg["max_insert_per_image"]]

            control_ids = match_inactive_insertion_controls(
                target_ids, active_ids, self.target_class, self.control_pool_ids,
                self.reference_median, self.decoder_norms, self.stats
            )
            n = min(target_ids.numel(), control_ids.numel())
            target_ids, control_ids = target_ids[:n], control_ids[:n]

            remove_ids = choose_source_removals(
                sparse_row, active_ids, self.target_class, self.stats, n
            )
            n = min(n, remove_ids.numel())
            target_ids, control_ids, remove_ids = target_ids[:n], control_ids[:n], remove_ids[:n]

            sparse_mod[0, remove_ids] = 0.0

            if self.mode == "target_insertion":
                values = self.reference_median[target_ids] * self.cfg["insertion_scale"]
                sparse_mod[0, target_ids] = values
                info["target_inserted_values"] = [float(x) for x in values.detach().cpu().tolist()]
            else:
                values = self.reference_median[control_ids] * self.cfg["insertion_scale"]
                sparse_mod[0, control_ids] = values
                info["control_inserted_values"] = [float(x) for x in values.detach().cpu().tolist()]

            info["target_feature_ids"] = [int(x) for x in target_ids.detach().cpu().tolist()]
            info["control_feature_ids"] = [int(x) for x in control_ids.detach().cpu().tolist()]
            info["removed_source_feature_ids"] = [int(x) for x in remove_ids.detach().cpu().tolist()]

        else:
            raise ValueError(f"Unknown intervention mode: {self.mode}")

        # Residual-preserving intervention:
        # original spatial feature + (edited SAE reconstruction - normal SAE reconstruction)
        recon = ksae_decode(sparse, self.ksae)
        recon_mod = ksae_decode(sparse_mod, self.ksae)
        delta = (recon_mod - recon)[:, :, None, None].to(dtype=x.dtype)
        y = x + delta

        info["edited_active_count"] = int(torch.count_nonzero(sparse_mod[0]).item())
        info["delta_l2"] = float(torch.linalg.vector_norm((recon_mod - recon)[0]).item())
        self.info = info
        return repack(y)


def encode_image_to_latent(pipe, image, dtype):
    image = image.to(device=pipe._execution_device, dtype=dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(image).latent_dist.mode()
    return latent * pipe.vae.config.scaling_factor


def add_noise(pipe, latent, timestep, seed):
    device, dtype = latent.device, latent.dtype
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)
    t = torch.tensor([timestep], dtype=torch.long, device=device)
    noisy = pipe.scheduler.add_noise(latent, noise, t)
    return noisy, noise


def predict_x0(pipe, noisy_latent, prompt_embeds, timestep):
    if pipe.scheduler.config.prediction_type != "epsilon":
        raise ValueError(
            f"This script expects epsilon prediction, got {pipe.scheduler.config.prediction_type}."
        )

    device, dtype = noisy_latent.device, noisy_latent.dtype
    t = torch.tensor([timestep], dtype=torch.long, device=device)
    model_input = pipe.scheduler.scale_model_input(noisy_latent, t)

    with torch.no_grad():
        noise_pred = pipe.unet(
            model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=True,
        ).sample

    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    alpha_t = alphas_cumprod[timestep]
    sqrt_alpha = alpha_t.sqrt().view(1, 1, 1, 1)
    sqrt_one_minus_alpha = (1.0 - alpha_t).sqrt().view(1, 1, 1, 1)
    return (noisy_latent - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha


def decode_latent(pipe, latent):
    latent = latent / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latent, return_dict=True).sample
    return ((image + 1.0) / 2.0).clamp(0.0, 1.0)


def save_tensor_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, path)


def get_source_indices(hf_dataset, cfg):
    explicit = cfg.get("source_indices", [])
    if explicit:
        return [int(x) for x in explicit]

    source_class = cfg.get("source_class")
    if source_class is None:
        raise ValueError("Set either source_indices or source_class.")

    labels = hf_dataset["label"]
    candidates = [i for i, label in enumerate(labels) if int(label) == int(source_class)]
    rng = random.Random(cfg["random_seed"])
    rng.shuffle(candidates)
    return candidates[:cfg["max_source_images"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    set_seed(cfg["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(cfg.get("torch_dtype", "float16" if torch.cuda.is_available() else "float32"))
    print("Device:", device, "| dtype:", dtype)

    # We only use Revelio's dataset helper here. The diffusion model itself is a normal
    # Stable Diffusion pipeline, not Revelio's feature-extraction UNet wrapper.
    sys.path.insert(0, cfg["diffc_dir"])
    sys.path.insert(1, cfg["sd_ksae_dir"])
    from constants import clip_transforms, diffusion_transformers_val
    from helpers.dataset import HuggingFaceImageDataset, load_huggingface_dataset

    print("Loading normal Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_name"],
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_pretrained(cfg["model_name"], subfolder="scheduler")
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.set_progress_bar_config(disable=True)

    prompt_embeds = encode_empty_prompt(pipe, device, dtype)
    hook_module = resolve_hook_module(pipe.unet, cfg["diffusion_layer"])
    print("Hook module:", cfg["diffusion_layer"], "|", hook_module.__class__.__name__)

    print("Loading k-SAE and feature statistics...")
    stats = load_feature_stats(cfg["feature_dir"], device)
    ksae = load_ksae(cfg["ksae_checkpoint_path"], device, default_k=cfg.get("ksae_k", 32))
    summary_df = pd.read_csv(cfg["feature_summary_path"])
    reference_median, reference_count = load_reference_cache(cfg["reference_activation_cache"], device)
    decoder_norms = torch.linalg.vector_norm(ksae["W_dec"], dim=1)

    target_feature_ids = select_high_purity_features(
        summary_df, cfg["target_class"], cfg["min_purity"], cfg["min_valid"]
    )
    control_pool_ids = build_control_pool(
        summary_df, reference_median, reference_count, cfg, device
    )

    print(f"Target class {cfg['target_class']}: {len(target_feature_ids)} qualifying features")
    print(f"Eligible insertion-control pool: {control_pool_ids.numel()} features")

    print("Loading dataset...")
    hf_dataset = load_huggingface_dataset(cfg["dataset_flag"], split=cfg.get("split", "test"))
    class_names = hf_dataset.features["label"].names
    dataset = HuggingFaceImageDataset(hf_dataset, diffusion_transformers_val, clip_transforms)
    source_indices = get_source_indices(hf_dataset, cfg)

    target_name = class_names[cfg["target_class"]]
    source_name = class_names[cfg["source_class"]] if cfg.get("source_class") is not None else "custom"
    print("Target:", target_name, "| Source:", source_name, "| images:", source_indices)

    output_dir = Path(cfg["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        "baseline",
        "target_ablation",
        "control_ablation",
        "target_amplification",
        "control_amplification",
        "target_insertion",
        "control_insertion",
    ]

    summary = {
        "target_class": cfg["target_class"],
        "target_class_name": target_name,
        "source_class": cfg.get("source_class"),
        "source_class_name": source_name,
        "source_indices": source_indices,
        "diffusion_timestep": cfg["diffusion_timestep"],
        "diffusion_layer": cfg["diffusion_layer"],
        "amplification_factor": cfg["amplification_factor"],
        "insertion_scale": cfg["insertion_scale"],
        "n_qualifying_target_features": len(target_feature_ids),
        "images": [],
    }

    for source_index in source_indices:
        diffusion_image, _, label, _ = dataset[source_index]
        source = diffusion_image.unsqueeze(0).to(device=device, dtype=dtype)
        source_vis = ((source.float().cpu()[0] + 1.0) / 2.0).clamp(0.0, 1.0)

        latent = encode_image_to_latent(pipe, source, dtype)
        noisy_latent, _ = add_noise(
            pipe, latent, cfg["diffusion_timestep"],
            seed=cfg["random_seed"] + int(source_index)
        )

        item_dir = output_dir / f"img_{source_index:05d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        save_tensor_image(source_vis, item_dir / "source.png")

        image_summary = {
            "source_index": int(source_index),
            "source_label": int(label),
            "source_label_name": class_names[int(label)],
            "conditions": {},
        }

        grid_images = [source_vis]

        for condition in conditions:
            if condition == "baseline":
                x0 = predict_x0(pipe, noisy_latent, prompt_embeds, cfg["diffusion_timestep"])
                hook_info = {}
            else:
                hook = SAEInterventionHook(
                    mode=condition,
                    ksae=ksae,
                    stats=stats,
                    target_feature_ids=target_feature_ids,
                    target_class=cfg["target_class"],
                    reference_median=reference_median,
                    reference_count=reference_count,
                    control_pool_ids=control_pool_ids,
                    decoder_norms=decoder_norms,
                    cfg=cfg,
                )
                handle = hook_module.register_forward_hook(hook)
                try:
                    x0 = predict_x0(pipe, noisy_latent, prompt_embeds, cfg["diffusion_timestep"])
                finally:
                    handle.remove()
                hook_info = hook.info

            recon = decode_latent(pipe, x0)[0].float().cpu()
            save_tensor_image(recon, item_dir / f"{condition}.png")
            grid_images.append(recon)
            image_summary["conditions"][condition] = hook_info

        baseline = grid_images[1]
        image_summary["baseline_pixel_mse_vs_source"] = float(
            torch.mean((baseline - source_vis) ** 2).item()
        )

        grid = make_grid(grid_images, nrow=len(grid_images), padding=2)
        save_tensor_image(grid, item_dir / "grid.png")
        save_json(image_summary, item_dir / "summary.json")
        summary["images"].append(image_summary)

        print(
            f"Finished index={source_index} label={class_names[int(label)]} "
            f"baseline_mse={image_summary['baseline_pixel_mse_vs_source']:.6f}"
        )

    save_json(summary, output_dir / "summary.json")
    print("Saved:", output_dir / "summary.json")


if __name__ == "__main__":
    main()