import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_torch_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(value):
    if value.lower() == "all":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def get_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if pd.isna(value) if not isinstance(value, (str, list, dict)) else False:
        return None
    return value


def load_ksae(path, device, default_k=32):
    checkpoint = torch.load(path, map_location=device)
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
    pre_acts = (x - ksae["b_dec"]) @ ksae["W_enc"] + ksae["b_enc"]
    top_values, top_indices = torch.topk(pre_acts, k=ksae["k"], dim=-1)
    sparse = torch.zeros_like(pre_acts)
    sparse.scatter_(dim=-1, index=top_indices, src=top_values)
    return sparse


def load_reference_cache(path, device, n_features):
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict) or "median" not in cache or "count" not in cache:
        raise KeyError("reference_activation_medians.pt must contain keys 'median' and 'count'.")

    median = cache["median"].float()
    count = cache["count"].long()
    if median.numel() != n_features or count.numel() != n_features:
        raise ValueError(
            f"Reference cache size mismatch: median={median.numel()}, count={count.numel()}, "
            f"k-SAE features={n_features}."
        )
    return median.to(device), count.to(device)


def build_diffc_model(args, device, input_channels, num_classes):
    from constants import model_base_dict
    from models import ImageClassifer

    config = {
        "dataset_flag": args.dataset_flag,
        "output_dir": "",
        "seed": args.seed,
        "model_name": args.model_name,
        "diffusion_timestep": args.diffusion_timestep,
        "diffusion_layer": args.diffusion_layer,
        "learning_rate": 1e-4,
        "num_epochs": 90,
        "batch_size": 1,
        "prompt_type": "empty",
        "pooling_strategy": "GAP",
        "dropout_rate": args.diffc_dropout_rate,
        "num_classes": num_classes,
        "num_devices": 1,
        "feature_model": model_base_dict[args.model_name],
        "diffusion_step_type": "onestep",
        "device": device,
        "input_channels": input_channels,
    }

    model = ImageClassifer(config).to(device)
    checkpoint = torch.load(args.diffc_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded Diff-C checkpoint epoch:", checkpoint.get("epoch"))
    return model


def compute_feature_label_stats(model, train_dataset, ksae, args, device):
    """
    Reconstruct majority-label / top-10 purity directly from the training set.
    No feature_summary_top10.csv or label-purity tensor files are required.

    As in the original k-SAE evaluation, only positive SAE activations are
    considered when collecting each feature's highest-activating images.
    """
    loader = DataLoader(
        train_dataset,
        batch_size=args.stats_batch_size,
        shuffle=False,
        pin_memory=False,
    )

    # Only features that ever receive positive activation are stored here.
    entries = {}
    n_seen = 0

    print("\nComputing top-10 class statistics from the training set...")
    set_seed(args.seed)

    with torch.no_grad():
        for batch_idx, (diffusion_images, _, labels, _) in enumerate(loader):
            diffusion_images = diffusion_images.to(device)
            labels = labels.to(device)
            features = model.get_features(diffusion_images, None, args.diffusion_timestep)
            sparse = ksae_encode(features.mean(dim=(2, 3)).float(), ksae)

            for i in range(sparse.size(0)):
                ids = torch.nonzero(sparse[i] > 0, as_tuple=False).flatten()
                if ids.numel() == 0:
                    continue

                vals = sparse[i, ids].detach().float().cpu().tolist()
                ids_cpu = ids.detach().cpu().tolist()
                label = int(labels[i].item())
                for fid, value in zip(ids_cpu, vals):
                    entries.setdefault(int(fid), []).append((float(value), label))

            n_seen += labels.size(0)
            if batch_idx % 25 == 0:
                print(f"  stats batch {batch_idx}/{len(loader)} | images={n_seen}")

            if args.stats_max_train_images > 0 and n_seen >= args.stats_max_train_images:
                break

    n_features = ksae["n_features"]
    majority_label = torch.full((n_features,), -1, dtype=torch.long, device=device)
    label_purity = torch.zeros((n_features,), dtype=torch.float32, device=device)
    valid_count = torch.zeros((n_features,), dtype=torch.long, device=device)

    for fid, feature_entries in entries.items():
        feature_entries.sort(key=lambda x: x[0], reverse=True)
        top = feature_entries[:10]
        if not top:
            continue

        labels = [label for _, label in top]
        values, counts = np.unique(labels, return_counts=True)
        pos = int(np.argmax(counts))
        majority_label[fid] = int(values[pos])
        label_purity[fid] = float(counts[pos] / len(labels))
        valid_count[fid] = len(labels)

    print(
        "Feature stats complete | features with >=1 valid top activation:",
        int((valid_count > 0).sum().item()),
        "| features with 10:",
        int((valid_count == 10).sum().item()),
    )

    return {
        "majority_label": majority_label,
        "label_purity": label_purity,
        "valid_count": valid_count,
    }


def build_feature_pools(stats, reference_median, reference_count, class_ids, args, device):
    finite_ref = ~torch.isnan(reference_median)
    base_eligible = (
        (stats["label_purity"] >= args.min_purity)
        & (stats["valid_count"] >= args.min_valid)
        & (reference_count >= args.min_reference_count)
        & finite_ref
    )

    target_ranked = {}
    for class_id in class_ids:
        ids = torch.nonzero(
            base_eligible & (stats["majority_label"] == int(class_id)),
            as_tuple=False,
        ).flatten()

        if ids.numel() == 0:
            target_ranked[int(class_id)] = ids
            continue

        if args.insertion_feature_ranking == "purity_reference_activation":
            score = stats["label_purity"][ids] * reference_median[ids].abs()
        elif args.insertion_feature_ranking == "reference_activation":
            score = reference_median[ids].abs()
        elif args.insertion_feature_ranking == "purity":
            score = stats["label_purity"][ids]
        else:
            raise ValueError(f"Unknown ranking: {args.insertion_feature_ranking}")

        target_ranked[int(class_id)] = ids[torch.argsort(score, descending=True)]

    control_eligible = (
        (stats["label_purity"] >= args.control_min_purity)
        & (stats["valid_count"] >= args.control_min_valid)
        & (reference_count >= args.min_reference_count)
        & finite_ref
    )
    control_pool = torch.nonzero(control_eligible, as_tuple=False).flatten().to(device)
    return target_ranked, control_pool


def match_inactive_controls(
    target_ids, active_ids, target_class, control_pool, reference_median,
    decoder_norms, stats
):
    if target_ids.numel() == 0:
        return target_ids

    active_mask = torch.zeros(reference_median.numel(), dtype=torch.bool, device=reference_median.device)
    active_mask[active_ids] = True
    pool = control_pool[~active_mask[control_pool]]
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

    return torch.stack(matched) if matched else control_pool[:0]


def choose_source_removals(sparse_row, active_ids, target_class, stats, n_remove):
    if n_remove <= 0:
        return active_ids[:0]

    pool = active_ids[stats["majority_label"][active_ids] != target_class]
    if pool.numel() == 0:
        return pool

    order = torch.argsort(sparse_row[pool].abs(), descending=False)
    return pool[order[:min(n_remove, pool.numel())]]


def margin_for_class(logits, class_id):
    target_logit = logits[class_id]
    others = logits.clone()
    others[class_id] = -torch.inf
    return target_logit - others.max()


def classifier_logits_for_deltas(model, feature_map, deltas, chunk_size):
    if not deltas:
        return torch.empty((0, model.classifer.fc.out_features), device=feature_map.device)

    stacked = torch.stack(deltas, dim=0).to(feature_map.device)
    outputs = []
    with torch.no_grad():
        for start in range(0, stacked.size(0), chunk_size):
            delta = stacked[start:start + chunk_size]
            batch_features = feature_map.expand(delta.size(0), -1, -1, -1) + delta[:, :, None, None]
            outputs.append(model.classifer(batch_features))
    return torch.cat(outputs, dim=0)


def evaluate_best_cases(
    model, test_dataset, test_labels, class_names, ksae, stats,
    reference_median, reference_count, target_ranked, control_pool,
    class_ids, args, device
):
    decoder_norms = torch.linalg.vector_norm(ksae["W_dec"], dim=1)
    counts = {int(c): 0 for c in class_ids}
    rows = []

    indices = list(range(len(test_dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    print("\nSearching for strong classifier-level intervention cases...")

    for scan_pos, image_index in enumerate(indices):
        remaining = [c for c in class_ids if counts[int(c)] < args.max_source_images_per_target]
        if not remaining:
            break

        diffusion_image, _, label, _ = test_dataset[image_index]
        true_label = int(label)
        source = diffusion_image.unsqueeze(0).to(device)

        image_seed = args.noise_seed + int(image_index)
        set_torch_seed(image_seed)

        with torch.no_grad():
            features = model.get_features(source, None, args.diffusion_timestep)
            pooled = features.mean(dim=(2, 3)).float()
            sparse = ksae_encode(pooled, ksae)
            sparse_row = sparse[0]
            active_ids = torch.nonzero(sparse_row != 0, as_tuple=False).flatten()
            logits_original = model.classifer(features)[0]

        original_pred = int(logits_original.argmax().item())
        if args.require_originally_correct and original_pred != true_label:
            continue

        candidate_targets = [
            int(c) for c in remaining
            if int(c) != true_label
            and int(c) != original_pred
            and target_ranked[int(c)].numel() > 0
        ]
        if not candidate_targets:
            continue

        specs = []
        valid_targets_this_image = set()

        for target_class in candidate_targets:
            ranked_ids = target_ranked[target_class]
            active_target = active_ids[torch.isin(active_ids, ranked_ids)]
            if args.require_no_target_features_active and active_target.numel() > 0:
                continue

            inactive_ids = ranked_ids[~torch.isin(ranked_ids, active_ids)]
            target_ids = inactive_ids[:args.max_insert_per_image]
            if target_ids.numel() == 0:
                continue

            control_ids = match_inactive_controls(
                target_ids, active_ids, target_class, control_pool,
                reference_median, decoder_norms, stats,
            )
            n = min(target_ids.numel(), control_ids.numel())
            if n == 0:
                continue

            target_ids = target_ids[:n]
            control_ids = control_ids[:n]
            remove_ids = choose_source_removals(
                sparse_row, active_ids, target_class, stats, n
            )
            n = min(n, remove_ids.numel())
            if n == 0:
                continue

            target_ids = target_ids[:n]
            control_ids = control_ids[:n]
            remove_ids = remove_ids[:n]
            removed_values = sparse_row[remove_ids]
            remove_delta = -(removed_values @ ksae["W_dec"][remove_ids])

            for scale in args.insertion_scales:
                target_values = reference_median[target_ids] * scale
                control_values = reference_median[control_ids] * scale

                target_delta = remove_delta + target_values @ ksae["W_dec"][target_ids]
                control_delta = remove_delta + control_values @ ksae["W_dec"][control_ids]

                specs.append({
                    "image_index": int(image_index),
                    "image_seed": int(image_seed),
                    "true_label": true_label,
                    "true_label_name": class_names[true_label],
                    "target_class": target_class,
                    "target_class_name": class_names[target_class],
                    "insertion_scale": float(scale),
                    "original_pred": original_pred,
                    "original_pred_name": class_names[original_pred],
                    "target_neuron_ids": [int(x) for x in target_ids.detach().cpu().tolist()],
                    "control_neuron_ids": [int(x) for x in control_ids.detach().cpu().tolist()],
                    "removed_source_neuron_ids": [int(x) for x in remove_ids.detach().cpu().tolist()],
                    "target_inserted_values": [float(x) for x in target_values.detach().cpu().tolist()],
                    "control_inserted_values": [float(x) for x in control_values.detach().cpu().tolist()],
                    "removed_source_values": [float(x) for x in removed_values.detach().cpu().tolist()],
                    "target_delta": target_delta,
                    "control_delta": control_delta,
                })
            valid_targets_this_image.add(target_class)

        if not specs:
            continue

        target_logits = classifier_logits_for_deltas(
            model, features, [s["target_delta"] for s in specs], args.classifier_chunk_size
        )
        control_logits = classifier_logits_for_deltas(
            model, features, [s["control_delta"] for s in specs], args.classifier_chunk_size
        )

        for i, spec in enumerate(specs):
            target_class = spec["target_class"]
            original_margin = float(margin_for_class(logits_original, target_class).item())
            targeted_margin = float(margin_for_class(target_logits[i], target_class).item())
            control_margin = float(margin_for_class(control_logits[i], target_class).item())

            targeted_pred = int(target_logits[i].argmax().item())
            control_pred = int(control_logits[i].argmax().item())
            original_target_logit = float(logits_original[target_class].item())
            targeted_target_logit = float(target_logits[i, target_class].item())
            control_target_logit = float(control_logits[i, target_class].item())

            row = {k: v for k, v in spec.items() if k not in {"target_delta", "control_delta"}}
            row.update({
                "targeted_pred": targeted_pred,
                "targeted_pred_name": class_names[targeted_pred],
                "control_pred": control_pred,
                "control_pred_name": class_names[control_pred],
                "original_target_margin": original_margin,
                "targeted_target_margin": targeted_margin,
                "control_target_margin": control_margin,
                "targeted_target_margin_increase": targeted_margin - original_margin,
                "control_target_margin_increase": control_margin - original_margin,
                "target_margin_increase_difference": targeted_margin - control_margin,
                "original_target_logit": original_target_logit,
                "targeted_target_logit": targeted_target_logit,
                "control_target_logit": control_target_logit,
                "target_logit_increase_difference": targeted_target_logit - control_target_logit,
                "target_flip": targeted_pred == target_class,
                "control_flip": control_pred == target_class,
                "target_crossed_zero": original_margin <= 0 and targeted_margin > 0,
                "control_crossed_zero": original_margin <= 0 and control_margin > 0,
            })
            rows.append(row)

        for target_class in valid_targets_this_image:
            counts[target_class] += 1

        if scan_pos % 10 == 0:
            minimum = min(counts.values()) if counts else 0
            maximum = max(counts.values()) if counts else 0
            print(
                f"  scanned={scan_pos + 1}/{len(indices)} | rows={len(rows)} | "
                f"per-target valid images={minimum}..{maximum}"
            )

    return rows, counts


def rank_best_cases(rows, args):
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["selection_priority"] = 3
    df.loc[df["target_crossed_zero"] & ~df["control_crossed_zero"], "selection_priority"] = 1
    df.loc[df["target_flip"], "selection_priority"] = 2
    df.loc[df["target_flip"] & ~df["control_flip"], "selection_priority"] = 0

    df = df[df["target_margin_increase_difference"] >= args.min_margin_difference].copy()
    if df.empty:
        return df

    df = df.sort_values(
        [
            "selection_priority",
            "target_margin_increase_difference",
            "targeted_target_margin_increase",
            "target_logit_increase_difference",
            "insertion_scale",
        ],
        ascending=[True, False, False, False, True],
    )

    # Keep only the strongest dose for each source-image / target-class pair.
    df = df.drop_duplicates(subset=["target_class", "image_index"], keep="first")

    if args.best_cases_per_target > 0:
        df = (
            df.groupby("target_class", group_keys=False)
            .head(args.best_cases_per_target)
            .reset_index(drop=True)
        )

    df = df.sort_values(
        ["selection_priority", "target_margin_increase_difference", "targeted_target_margin_increase"],
        ascending=[True, False, False],
    )

    if args.top_n > 0:
        df = df.head(args.top_n)
    return df.reset_index(drop=True)


# -------------------------- Generation helpers --------------------------

def encode_empty_prompt(pipe, device, dtype):
    tokens = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeds = pipe.text_encoder(tokens.input_ids.to(device))[0]
    return embeds.to(device=device, dtype=dtype)


def resolve_hook_module(unet, diffusion_layer):
    if diffusion_layer.startswith("up_ft:"):
        idx = int(diffusion_layer.split(":")[1])
        return unet.up_blocks[idx]
    if diffusion_layer in {"mid", "bottleneck:0"}:
        return unet.mid_block
    raise ValueError("Generation supports diffusion_layer='up_ft:x', 'mid', or 'bottleneck:0'.")


def unpack_hook_output(output):
    if torch.is_tensor(output):
        return output, lambda new_x: new_x
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Unsupported tuple hook output.")
        return output[0], lambda new_x: (new_x,) + output[1:]
    if isinstance(output, list):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Unsupported list hook output.")
        return output[0], lambda new_x: [new_x] + output[1:]
    raise TypeError(f"Unsupported hook output: {type(output)}")


class FixedInsertionHook:
    """Apply the exact feature IDs and inserted values from a selected Diff-C case."""

    def __init__(self, mode, ksae, row):
        self.mode = mode
        self.ksae = ksae
        self.row = row
        self.info = {}

    def __call__(self, module, inputs, output):
        x, repack = unpack_hook_output(output)
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f"Expected hooked feature [1,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.ksae["d_in"]:
            raise ValueError(
                f"Hook channels {x.shape[1]} do not match k-SAE d_in={self.ksae['d_in']}."
            )

        pooled = x.float().mean(dim=(2, 3))
        sparse = ksae_encode(pooled, self.ksae)
        sparse_mod = sparse.clone()

        remove_ids = torch.tensor(
            self.row["removed_source_neuron_ids"], dtype=torch.long, device=x.device
        )
        sparse_mod[0, remove_ids] = 0.0

        if self.mode == "target":
            ids = torch.tensor(self.row["target_neuron_ids"], dtype=torch.long, device=x.device)
            values = torch.tensor(self.row["target_inserted_values"], dtype=torch.float32, device=x.device)
        elif self.mode == "control":
            ids = torch.tensor(self.row["control_neuron_ids"], dtype=torch.long, device=x.device)
            values = torch.tensor(self.row["control_inserted_values"], dtype=torch.float32, device=x.device)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        sparse_mod[0, ids] = values
        recon = sparse @ self.ksae["W_dec"] + self.ksae["b_dec"]
        recon_mod = sparse_mod @ self.ksae["W_dec"] + self.ksae["b_dec"]
        delta = (recon_mod - recon)[:, :, None, None].to(dtype=x.dtype)

        self.info = {
            "mode": self.mode,
            "feature_ids": [int(v) for v in ids.detach().cpu().tolist()],
            "removed_source_neuron_ids": [int(v) for v in remove_ids.detach().cpu().tolist()],
            "inserted_values": [float(v) for v in values.detach().cpu().tolist()],
            "delta_l2": float(torch.linalg.vector_norm((recon_mod - recon)[0]).item()),
        }
        return repack(x + delta)


def encode_image_to_latent(pipe, image, dtype):
    image = image.to(device=pipe._execution_device, dtype=dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(image).latent_dist.mode()
    return latent * pipe.vae.config.scaling_factor


def add_noise(pipe, latent, timestep, seed):
    generator = torch.Generator(device=latent.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        latent.shape,
        generator=generator,
        device=latent.device,
        dtype=latent.dtype,
    )
    t = torch.tensor([timestep], dtype=torch.long, device=latent.device)
    return pipe.scheduler.add_noise(latent, noise, t)


def predict_noise(pipe, latent, prompt_embeds, timestep):
    if pipe.scheduler.config.prediction_type != "epsilon":
        raise ValueError(f"Expected epsilon prediction, got {pipe.scheduler.config.prediction_type}")
    t = torch.tensor([timestep], dtype=torch.long, device=latent.device)
    model_input = pipe.scheduler.scale_model_input(latent, t)
    with torch.no_grad():
        return pipe.unet(
            model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=True,
        ).sample


def ddim_step_contiguous(pipe, sample, noise_pred, timestep):
    alphas = pipe.scheduler.alphas_cumprod.to(device=sample.device, dtype=torch.float32)
    sample_f = sample.float()
    noise_f = noise_pred.float()
    alpha_t = alphas[timestep]
    alpha_prev = alphas[timestep - 1] if timestep > 0 else torch.tensor(
        1.0, device=sample.device, dtype=torch.float32
    )
    pred_x0 = (sample_f - torch.sqrt(1.0 - alpha_t) * noise_f) / torch.sqrt(alpha_t)
    prev = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * noise_f
    return prev.to(dtype=sample.dtype)


def denoise_multistep(pipe, noisy_latent, prompt_embeds, start_timestep, hook_module=None, hook=None):
    latent = noisy_latent.clone()
    hook_info = {}

    for timestep in range(int(start_timestep), -1, -1):
        handle = None
        if timestep == int(start_timestep) and hook is not None:
            handle = hook_module.register_forward_hook(hook)

        try:
            noise_pred = predict_noise(pipe, latent, prompt_embeds, timestep)
        finally:
            if handle is not None:
                handle.remove()

        if timestep == int(start_timestep) and hook is not None:
            hook_info = dict(hook.info)

        latent = ddim_step_contiguous(pipe, latent, noise_pred, timestep)
    return latent, hook_info


def decode_latent(pipe, latent):
    latent = latent / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latent, return_dict=True).sample
    return ((image + 1.0) / 2.0).clamp(0.0, 1.0)


def save_tensor_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, path)


def generate_best_cases(best_df, test_dataset, class_names, ksae, args, device):
    if best_df.empty:
        print("No best cases found; skipping generation.")
        return

    dtype = get_dtype(args.generation_dtype)
    print("\nLoading Stable Diffusion pipeline for qualitative generation...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.set_progress_bar_config(disable=True)

    prompt_embeds = encode_empty_prompt(pipe, device, dtype)
    hook_module = resolve_hook_module(pipe.unet, args.diffusion_layer)
    generation_root = Path(args.output_dir) / "generated_best_cases"
    generation_root.mkdir(parents=True, exist_ok=True)
    generation_summary = []

    for rank, row in best_df.iterrows():
        row = row.to_dict()
        image_index = int(row["image_index"])
        target_class = int(row["target_class"])
        source_image, _, source_label, _ = test_dataset[image_index]
        source = source_image.unsqueeze(0).to(device=device, dtype=dtype)
        source_vis = ((source.float().cpu()[0] + 1.0) / 2.0).clamp(0.0, 1.0)

        latent = encode_image_to_latent(pipe, source, dtype)
        noisy_latent = add_noise(pipe, latent, args.diffusion_timestep, int(row["image_seed"]))

        baseline_latent, _ = denoise_multistep(
            pipe, noisy_latent, prompt_embeds, args.diffusion_timestep
        )
        target_hook = FixedInsertionHook("target", ksae, row)
        target_latent, target_info = denoise_multistep(
            pipe, noisy_latent, prompt_embeds, args.diffusion_timestep,
            hook_module=hook_module, hook=target_hook,
        )
        control_hook = FixedInsertionHook("control", ksae, row)
        control_latent, control_info = denoise_multistep(
            pipe, noisy_latent, prompt_embeds, args.diffusion_timestep,
            hook_module=hook_module, hook=control_hook,
        )

        baseline = decode_latent(pipe, baseline_latent)[0].float().cpu()
        targeted = decode_latent(pipe, target_latent)[0].float().cpu()
        control = decode_latent(pipe, control_latent)[0].float().cpu()
        target_diff = torch.abs(targeted - baseline)
        control_diff = torch.abs(control - baseline)

        target_name = class_names[target_class].replace(" ", "_")
        source_name = class_names[int(source_label)].replace(" ", "_")
        case_dir = generation_root / (
            f"{rank + 1:02d}_idx{image_index}_{source_name}_to_{target_name}_"
            f"scale{float(row['insertion_scale']):g}"
        )
        case_dir.mkdir(parents=True, exist_ok=True)

        save_tensor_image(source_vis, case_dir / "source.png")
        save_tensor_image(baseline, case_dir / "baseline.png")
        save_tensor_image(targeted, case_dir / "target_insertion.png")
        save_tensor_image(control, case_dir / "control_insertion.png")
        save_tensor_image(target_diff, case_dir / "target_diff.png")
        save_tensor_image(control_diff, case_dir / "control_diff.png")
        save_tensor_image(
            torch.clamp(target_diff * args.difference_map_scale, 0, 1),
            case_dir / "target_diff_scaled.png",
        )
        save_tensor_image(
            torch.clamp(control_diff * args.difference_map_scale, 0, 1),
            case_dir / "control_diff_scaled.png",
        )

        grid = make_grid(
            [
                source_vis,
                baseline,
                targeted,
                control,
                torch.clamp(target_diff * args.difference_map_scale, 0, 1),
                torch.clamp(control_diff * args.difference_map_scale, 0, 1),
            ],
            nrow=6,
            padding=2,
        )
        save_tensor_image(grid, case_dir / "grid.png")

        case_summary = {
            "classifier_case": jsonable(row),
            "source_label": int(source_label),
            "source_label_name": class_names[int(source_label)],
            "target_hook": target_info,
            "control_hook": control_info,
            "target_pixel_mae_vs_baseline": float(target_diff.mean().item()),
            "target_pixel_mse_vs_baseline": float(((targeted - baseline) ** 2).mean().item()),
            "control_pixel_mae_vs_baseline": float(control_diff.mean().item()),
            "control_pixel_mse_vs_baseline": float(((control - baseline) ** 2).mean().item()),
        }
        save_json(case_summary, case_dir / "summary.json")
        generation_summary.append(case_summary)

        print(
            f"Generated {rank + 1}/{len(best_df)} | idx={image_index} "
            f"{class_names[int(source_label)]} -> {class_names[target_class]} | "
            f"target_flip={bool(row['target_flip'])} | control_flip={bool(row['control_flip'])}"
        )

    save_json(generation_summary, generation_root / "summary.json")


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "End-to-end search for strong SAE insertion cases followed by qualitative "
            "Stable-Diffusion generation. The only precomputed analysis input is "
            "reference_activation_medians.pt; class-specific feature statistics are "
            "recomputed directly from the training set."
        )
    )

    # Core repositories / checkpoints.
    p.add_argument("--diffc-dir", required=True)
    p.add_argument("--sd-ksae-dir", required=True)
    p.add_argument("--diffc-checkpoint", required=True)
    p.add_argument("--ksae-checkpoint", required=True)
    p.add_argument("--reference-cache", required=True, help="reference_activation_medians.pt")
    p.add_argument("--output-dir", required=True)

    # Dataset / model.
    p.add_argument("--dataset-flag", default="timm/oxford-iiit-pet")
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--model-name", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--diffusion-timestep", type=int, default=25)
    p.add_argument("--diffusion-layer", default="up_ft:1")
    p.add_argument("--diffc-dropout-rate", type=float, default=0.5)
    p.add_argument("--ksae-k", type=int, default=32)

    # Dynamic class-specific feature-stat computation.
    p.add_argument("--stats-batch-size", type=int, default=8)
    p.add_argument(
        "--stats-max-train-images",
        type=int,
        default=0,
        help="0 means use the full training split.",
    )
    p.add_argument("--min-purity", type=float, default=0.8)
    p.add_argument("--min-valid", type=int, default=10)
    p.add_argument("--control-min-purity", type=float, default=0.8)
    p.add_argument("--control-min-valid", type=int, default=10)
    p.add_argument("--min-reference-count", type=int, default=5)
    p.add_argument(
        "--insertion-feature-ranking",
        choices=["purity_reference_activation", "reference_activation", "purity"],
        default="purity_reference_activation",
    )

    # Best-case search.
    p.add_argument("--target-classes", default="all", help="all or comma-separated class IDs")
    p.add_argument("--max-source-images-per-target", type=int, default=64)
    p.add_argument("--max-insert-per-image", type=int, default=3)
    p.add_argument("--insertion-scales", default="0.5,1.0,1.5,2.0")
    p.add_argument("--classifier-chunk-size", type=int, default=8)
    p.add_argument("--require-originally-correct", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-no-target-features-active", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-margin-difference", type=float, default=0.0)
    p.add_argument("--best-cases-per-target", type=int, default=2)
    p.add_argument("--top-n", type=int, default=10)

    # Generation.
    p.add_argument("--generation-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--difference-map-scale", type=float, default=10.0)

    # Reproducibility.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    args.insertion_scales = parse_float_list(args.insertion_scales)
    requested_target_classes = parse_int_list(args.target_classes)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    sys.path.insert(0, args.diffc_dir)
    sys.path.insert(1, args.sd_ksae_dir)
    from constants import clip_transforms, diffusion_transformers_val
    from helpers.dataset import HuggingFaceImageDataset, load_huggingface_dataset

    print("Loading dataset...")
    hf_train = load_huggingface_dataset(args.dataset_flag, split=args.train_split)
    hf_test = load_huggingface_dataset(args.dataset_flag, split=args.test_split)
    class_names = hf_test.features["label"].names
    num_classes = len(class_names)

    if requested_target_classes is None:
        class_ids = list(range(num_classes))
    else:
        class_ids = requested_target_classes
        invalid = [c for c in class_ids if c < 0 or c >= num_classes]
        if invalid:
            raise ValueError(f"Invalid target class IDs: {invalid}")

    train_dataset = HuggingFaceImageDataset(hf_train, diffusion_transformers_val, clip_transforms)
    test_dataset = HuggingFaceImageDataset(hf_test, diffusion_transformers_val, clip_transforms)
    test_labels = [int(x) for x in hf_test["label"]]

    print("Loading k-SAE...")
    ksae = load_ksae(args.ksae_checkpoint, device, args.ksae_k)
    print("k-SAE:", ksae["n_features"], "features | d_in=", ksae["d_in"], "| k=", ksae["k"])

    reference_median, reference_count = load_reference_cache(
        args.reference_cache, device, ksae["n_features"]
    )
    print(
        "Reference cache loaded | finite medians:",
        int((~torch.isnan(reference_median)).sum().item()),
        "| count>0:",
        int((reference_count > 0).sum().item()),
    )

    print("Loading Diff-C + diffusion feature extractor...")
    diffc_model = build_diffc_model(args, device, ksae["d_in"], num_classes)

    stats = compute_feature_label_stats(diffc_model, train_dataset, ksae, args, device)
    target_ranked, control_pool = build_feature_pools(
        stats, reference_median, reference_count, class_ids, args, device
    )

    print("\nEligible class-specific features:")
    for c in class_ids:
        print(f"  {c:02d} {class_names[c]}: {target_ranked[c].numel()}")
    print("Control pool:", control_pool.numel())

    rows, counts = evaluate_best_cases(
        diffc_model, test_dataset, test_labels, class_names, ksae, stats,
        reference_median, reference_count, target_ranked, control_pool,
        class_ids, args, device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results_path = output_dir / "all_candidate_interventions.json"
    save_json([jsonable(r) for r in rows], all_results_path)

    best_df = rank_best_cases(rows, args)
    best_csv = output_dir / "best_cases.csv"
    best_json = output_dir / "best_cases.json"
    best_df.to_csv(best_csv, index=False)
    save_json([jsonable(r) for r in best_df.to_dict(orient="records")], best_json)

    print("\nBest cases:")
    if best_df.empty:
        print("  None found with the current thresholds.")
    else:
        display_cols = [
            "true_label_name", "target_class_name", "image_index", "insertion_scale",
            "target_flip", "control_flip", "original_target_margin",
            "targeted_target_margin", "control_target_margin",
            "target_margin_increase_difference",
        ]
        print(best_df[display_cols].to_string(index=False))

    print("\nSaved classifier search results:")
    print(" ", all_results_path)
    print(" ", best_csv)
    print(" ", best_json)

    # Free the heavy Diff-C model before loading a normal SD pipeline for generation.
    del diffc_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generate_best_cases(best_df, test_dataset, class_names, ksae, args, device)
    print("\nDone. Outputs saved under:", output_dir)


if __name__ == "__main__":
    main()
