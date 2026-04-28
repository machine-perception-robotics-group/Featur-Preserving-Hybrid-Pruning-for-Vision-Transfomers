#!/usr/bin/env python3
"""
Compute cosine similarity between CLS token features of two checkpoints during evaluation.

The script runs inference on the validation (or synthetic) data loader for both checkpoints,
captures the activations that enter the classifier head (i.e., the normalized CLS token / pre-logits),
and reports cosine statistics per sample, aggregated averages, and optional per-class summaries.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import build_dataset
from main import infer_structured_mlp_dims, load_checkpoint
from models.dyvit import VisionTransformerDiffPruning
import utils


def parse_args():
    parser = argparse.ArgumentParser("CLS token cosine similarity evaluator")
    # Data-related arguments (mirroring main/infer defaults where needed)
    parser.add_argument(
        "--data-set",
        default="CIFAR",
        choices=["CIFAR", "IMNET", "image_folder", "STANFORD_DOGS"],
        help="Dataset type for evaluation.",
    )
    parser.add_argument("--data-path", default="", help="Root path to the dataset.")
    parser.add_argument("--eval-data-path", default=None,
                        help="Evaluation path when using image_folder datasets.")
    parser.add_argument("--nb-classes", type=int, default=1000)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--pin-mem", action="store_true", default=True)
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.add_argument("--crop-pct", type=float, default=None)
    parser.add_argument("--imagenet-default-mean-and-std", type=utils.str2bool, default=True)
    parser.add_argument("--color-jitter", type=float, default=0.4)
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--train-interpolation", type=str, default="bicubic")

    # Model instantiation args
    parser.add_argument("--model", default="deit-b", choices=["deit-b", "deit-s", "deit-256"],
                        help="Currently supported ViT variants for CLS token comparison.")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--base-rate", type=float, default=0.7,
                        help="Base keep ratio for the AFTER model when --token-rate is not provided.")
    parser.add_argument("--token-rate", type=float, nargs=3, default=None,
                        help="Explicit AFTER-model keep ratio (0-1) for each pruning stage.")
    parser.add_argument(
        "--extra-token-metrics",
        action="store_true",
        help="Compute additional L1/L2/MSE statistics for CLS / patch tokens (requires 'norm' capture).",
    )
    parser.add_argument(
        "--cls-metric-point",
        default="norm",
        help="Capture point name that represents the CLS token for extra metrics (default: norm).",
    )
    parser.add_argument(
        "--capture-points",
        nargs="+",
        default=["final"],
        help=(
            "Where to tap CLS features. "
            "Use 'final' for classifier input, 'pre_logits', 'norm', 'block{idx}' (post block), "
            "'block{idx}_pre' (block input / post-pruning), or 'prune{stage}' (stage order in pruning_loc). "
            "Multiple points can be specified."
        ),
    )
    parser.add_argument(
        "--capture-all-blocks",
        action="store_true",
        help="Automatically include block0..block{depth-1} CLS capture points.",
    )

    # Checkpoint handling
    parser.add_argument("--before-checkpoint", required=True, help="Path to the pre-pruning checkpoint.")
    parser.add_argument("--after-checkpoint", required=True, help="Path to the post-pruning checkpoint.")
    parser.add_argument("--before-key", default="model|model_ema|module|state_dict",
                        help="Candidate keys to locate the model weights inside the pre-pruning checkpoint.")
    parser.add_argument("--after-key", default="model|model_ema|module|state_dict",
                        help="Candidate keys to locate the model weights inside the post-pruning checkpoint.")
    parser.add_argument("--before-prefix", default="", help="Prefix to strip from pre-pruning state dict keys.")
    parser.add_argument("--after-prefix", default="", help="Prefix to strip from post-pruning state dict keys.")

    # Evaluation control
    parser.add_argument("--device", default="cuda", help="cuda or cpu device string.")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Limit the number of evaluation samples (-1 to use the full dataset).")
    parser.add_argument("--synthetic-samples", type=int, default=0,
                        help="If >0, bypass dataset loading and run on random inputs for quick debugging.")
    parser.add_argument("--per-class-topk", type=int, default=5,
                        help="Report the classes with the lowest average cosine similarity.")
    parser.add_argument(
        "--show-class-summary",
        action="store_true",
        help="Print per-class cosine summaries (off by default).",
    )
    parser.add_argument("--output-json", default="",
                        help="Optional path to dump per-sample cosine similarities.")
    parser.add_argument(
        "--dump-token-details",
        default="",
        help=(
            "If set, store per-sample pruning masks/scores/final tokens from the AFTER model into this JSON file. "
            "Use together with --max-samples to limit size."
        ),
    )
    parser.add_argument(
        "--report-pruning-impact",
        action="store_true",
        help=(
            "Summarize how many tokens survive each pruning stage for the lowest-similarity samples. "
            "Automatically enables token detail collection."
        ),
    )
    parser.add_argument(
        "--impact-reference",
        default="final",
        help="Capture point name that determines which samples are considered low-similarity.",
    )
    parser.add_argument(
        "--impact-bottom-k",
        type=int,
        default=16,
        help="Number of lowest-similarity samples (per reference capture) to inspect when reporting pruning impact.",
    )
    return parser.parse_args()


def build_vit_model(args, structured_mlp_dims, keep_all_tokens: bool = False):
    custom_rate = list(args.token_rate) if args.token_rate is not None else None
    if args.model == "deit-s":
        pruning_loc = [3, 6, 9]
        embed_dim = 384
        num_heads = 6
    elif args.model == "deit-b":
        pruning_loc = [3, 6, 9]
        embed_dim = 768
        num_heads = 12
    elif args.model == "deit-256":
        pruning_loc = [3, 6, 9]
        embed_dim = 256
        num_heads = 4
    else:
        raise ValueError(f"Unsupported model {args.model} for CLS token comparison.")

    if keep_all_tokens:
        keep_rate = [1.0] * len(pruning_loc)
    else:
        keep_rate = custom_rate if custom_rate is not None else [
            args.base_rate,
            args.base_rate ** 2,
            args.base_rate ** 3,
        ]

    model = VisionTransformerDiffPruning(
        patch_size=16,
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        pruning_loc=pruning_loc,
        token_ratio=keep_rate,
        distill=True,
        drop_path_rate=args.drop_path,
        num_classes=args.nb_classes,
        mlp_hidden_dims=structured_mlp_dims,
    )
    return model


def select_state_dict_blob(checkpoint: Dict, key_expr: str) -> Dict:
    if isinstance(checkpoint, dict):
        for candidate in key_expr.split("|"):
            key = candidate.strip()
            if key and key in checkpoint:
                blob = checkpoint[key]
                if isinstance(blob, dict):
                    return blob.copy()
    if isinstance(checkpoint, dict):
        return checkpoint.copy()
    raise TypeError("Checkpoint does not contain a compatible state dict.")


def maybe_strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    prefix = (prefix or "").strip()
    if not prefix:
        return state_dict
    if not prefix.endswith("."):
        prefix += "."
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, key_expr: str, prefix: str):
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = maybe_strip_prefix(select_state_dict_blob(checkpoint, key_expr), prefix)

    # Remove classifier if dimensions mismatch.
    model_state = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in state_dict and key in model_state and state_dict[key].shape != model_state[key].shape:
            print(f"[Warning] Removing incompatible key {key} from checkpoint {checkpoint_path}.")
            del state_dict[key]
    utils.load_state_dict(model, state_dict)


@dataclass
class FeatureHook:
    module: torch.nn.Module
    use_pre_hook: bool

    def __post_init__(self):
        self.data = None
        if self.use_pre_hook:
            self.handle = self.module.register_forward_pre_hook(self._pre_hook)
        else:
            self.handle = self.module.register_forward_hook(self._post_hook)

    def _pre_hook(self, module, inputs):
        self.data = inputs[0].detach()

    def _post_hook(self, module, inputs, output):
        self.data = output.detach()

    def pop(self) -> torch.Tensor:
        if self.data is None:
            raise RuntimeError("Hook did not capture any data this iteration.")
        value = self.data
        self.data = None
        return value

    def close(self):
        self.handle.remove()


@dataclass
class CaptureSpec:
    name: str
    hook: FeatureHook
    reducer: Callable[[torch.Tensor], torch.Tensor]


def _cls_from_sequence(values: torch.Tensor) -> torch.Tensor:
    if values.dim() != 3:
        raise ValueError(f"Expected a 3D tensor for CLS extraction, received shape {tuple(values.shape)}")
    return values[:, 0, :]


def _identity(values: torch.Tensor) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, received shape {tuple(values.shape)}")
    return values


def _l2_normalize(values: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return values / (values.norm(p=2, dim=dim, keepdim=True) + eps)


def _pairwise_cosine_matrix(values: torch.Tensor) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError("Pairwise cosine expects a 2D tensor (batch, dim).")
    normalized = F.normalize(values, dim=-1)
    return normalized @ normalized.t()


def _linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = y.t() @ x
    numerator = xty.norm(p="fro") ** 2
    denominator = (x.t() @ x).norm(p="fro") * (y.t() @ y).norm(p="fro") + eps
    return numerator / denominator


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    numerator = (x * y).sum()
    denominator = torch.sqrt(x.square().sum() * y.square().sum() + eps)
    return numerator / denominator


def _nn_at_k(sim_a: torch.Tensor, sim_b: torch.Tensor, k: int = 5) -> torch.Tensor:
    if sim_a.dim() != 2 or sim_b.dim() != 2:
        raise ValueError("NN@k expects square similarity matrices.")
    total = sim_a.shape[0]
    if total <= 1:
        return torch.tensor(0.0, device=sim_a.device)
    k = max(1, min(k, total - 1))
    mask = torch.eye(total, dtype=torch.bool, device=sim_a.device)
    sim_a = sim_a.masked_fill(mask, float("-inf"))
    sim_b = sim_b.masked_fill(mask, float("-inf"))
    idx_a = torch.topk(sim_a, k=k, dim=-1).indices
    idx_b = torch.topk(sim_b, k=k, dim=-1).indices.unsqueeze(1)
    matches = (idx_a.unsqueeze(-1) == idx_b).any(dim=-1).float().mean(dim=-1)
    return matches.mean()


def _split_block_capture(name: str) -> Tuple[str, bool]:
    suffixes = ("_pre", ":pre", "-pre", ".pre")
    capture_pre = False
    base = name
    for suffix in suffixes:
        if base.endswith(suffix):
            capture_pre = True
            base = base[: -len(suffix)]
            break
    return base, capture_pre


def create_capture_specs(model: torch.nn.Module, names: List[str]) -> List[CaptureSpec]:
    specs: List[CaptureSpec] = []
    pruning_loc = getattr(model, "pruning_loc", None)
    for raw_name in names:
        name = raw_name.lower()
        if name in {"final", "head", "cls"}:
            hook = FeatureHook(model.head, use_pre_hook=True)
            reducer = _identity
            label = raw_name
        elif name in {"pre_logits", "prelogits"}:
            if not hasattr(model, "pre_logits"):
                raise ValueError("Model missing pre_logits; cannot attach capture point.")
            hook = FeatureHook(model.pre_logits, use_pre_hook=False)
            reducer = _identity
            label = raw_name
        elif name in {"norm", "encoder_norm"}:
            hook = FeatureHook(model.norm, use_pre_hook=False)
            reducer = _cls_from_sequence
            label = raw_name
        elif name.startswith("block"):
            block_name, use_pre_hook = _split_block_capture(name)
            idx_str = block_name.replace("block", "")
            if not idx_str.isdigit():
                raise ValueError(f"Invalid block specification: {raw_name}")
            idx = int(idx_str)
            if idx < 0 or idx >= len(model.blocks):
                raise ValueError(f"Block index {idx} out of range (0-{len(model.blocks)-1}).")
            hook = FeatureHook(model.blocks[idx], use_pre_hook=use_pre_hook)
            reducer = _cls_from_sequence
            label = raw_name
        elif name.startswith("prune"):
            if pruning_loc is None:
                raise ValueError("Model does not expose pruning_loc; cannot use 'prune{idx}' capture points.")
            stage_str = name.replace("prune", "", 1)
            if not stage_str.isdigit():
                raise ValueError(f"Invalid prune stage specification: {raw_name}")
            stage_idx = int(stage_str)
            if stage_idx < 0 or stage_idx >= len(pruning_loc):
                raise ValueError(
                    f"Prune stage {stage_idx} out of range (0-{len(pruning_loc)-1})."
                )
            block_idx = pruning_loc[stage_idx]
            hook = FeatureHook(model.blocks[block_idx], use_pre_hook=True)
            reducer = _cls_from_sequence
            label = raw_name
        else:
            raise ValueError(f"Unsupported capture point '{raw_name}'.")
        specs.append(CaptureSpec(label, hook, reducer))
    return specs


def close_capture_specs(specs: List[CaptureSpec]):
    for spec in specs:
        spec.hook.close()


def iter_synthetic_batches(total_samples: int, batch_size: int, input_size: int):
    generated = 0
    while generated < total_samples:
        current = min(batch_size, total_samples - generated)
        images = torch.randn(current, 3, input_size, input_size)
        targets = torch.full((current,), -1, dtype=torch.long)
        yield images, targets
        generated += current


def cosine_similarity_distribution(
    model_before: torch.nn.Module,
    model_after: torch.nn.Module,
    data_iter: Iterable,
    device: torch.device,
    capture_points: List[str],
    max_samples: Optional[int] = None,
    collect_token_details: bool = False,
    extra_metrics: bool = False,
    cls_metric_point: str = "norm",
    patch_metrics_available: bool = True
):
    specs_before = create_capture_specs(model_before, capture_points)
    specs_after = create_capture_specs(model_after, capture_points)
    results: Dict[str, List[Dict]] = {spec.name: [] for spec in specs_before}
    token_records: List[Dict] = []
    metric_buffer: Optional[Dict[str, Dict[str, List[float]]]] = None
    cls_metric_key = cls_metric_point.lower()
    patch_warning_printed = False
    if extra_metrics:
        metric_buffer = {
            "cls": {"norm_l2": [], "norm_l1": [], "mse": [], "norm_l1_diff": []},
            "patch": {"cos": [], "norm_l2": [], "norm_l1": [], "mse": [], "norm_l1_diff": []},
            "cls_relation": {"mse": [], "linear_cka": [], "pearson": [], "nn_at5": []},
        }
    total = 0
    try:
        with torch.no_grad():
            for batch_idx, (images, targets) in tqdm(enumerate(data_iter)):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if max_samples is not None:
                    remaining = max_samples - total
                    if remaining <= 0:
                        break
                    if images.shape[0] > remaining:
                        images = images[:remaining]
                        targets = targets[:remaining]

                need_details = collect_token_details or extra_metrics
                if need_details:
                    logits_before, before_details = model_before(images, return_token_details=True)
                    logits_after, after_details = model_after(images, return_token_details=True)
                else:
                    logits_before = model_before(images)
                    logits_after = model_after(images)
                    before_details = None
                    after_details = None
                _ = logits_before, logits_after

                feats_before_raw = {
                    spec.name: spec.reducer(spec.hook.pop()).detach()
                    for spec in specs_before
                }
                feats_after_raw = {
                    spec.name: spec.reducer(spec.hook.pop()).detach()
                    for spec in specs_after
                }
                feats_before = {
                    name: F.normalize(value, dim=1)
                    for name, value in feats_before_raw.items()
                }
                feats_after = {
                    name: F.normalize(value, dim=1)
                    for name, value in feats_after_raw.items()
                }

                for name in results.keys():
                    cos = (feats_before[name] * feats_after[name]).sum(dim=1).clamp(min=-1.0, max=1.0)
                    raw_before = feats_before_raw[name]
                    raw_after = feats_after_raw[name]
                    cka_value = None
                    if extra_metrics and raw_before.size(0) >= 2:
                        try:
                            cka_value = float(_linear_cka(raw_before, raw_after).item())
                        except Exception:
                            cka_value = None
                    for idx in range(images.shape[0]):
                        target = int(targets[idx].item()) if targets[idx].item() >= 0 else None
                        entry = {
                            "index": total + idx,
                            "target": target,
                            "cosine": float(cos[idx].item()),
                        }
                        if extra_metrics:
                            diff = raw_before[idx] - raw_after[idx]
                            entry["mse"] = float(diff.pow(2).mean().item())
                            entry["norm_l2"] = float(diff.norm(p=2).item())
                            entry["norm_l1"] = float(diff.norm(p=1).item())
                            entry["norm_l1_diff"] = float(raw_after[idx].norm(p=1).item()) - float(raw_before[idx].norm(p=1).item())
                            if cka_value is not None and math.isfinite(cka_value):
                                entry["cka"] = cka_value
                        results[name].append(entry)

                if after_details is not None:
                    kept_masks = after_details.get("kept_masks", [])
                    score_history = after_details.get("scores", [])
                    tokens = after_details.get("tokens")
                    for idx in range(images.shape[0]):
                        record = {
                            "index": total + idx,
                            "target": int(targets[idx].item()) if targets[idx].item() >= 0 else None,
                        }
                        if kept_masks:
                            record["kept_masks"] = [stage[idx].detach().cpu().tolist() for stage in kept_masks]
                        if score_history:
                            record["scores"] = [stage[idx].detach().cpu().tolist() for stage in score_history]
                        if tokens is not None:
                            record["tokens"] = tokens[idx].detach().cpu().tolist()
                        token_records.append(record)

                if extra_metrics:
                    if cls_metric_key not in feats_before_raw or cls_metric_key not in feats_after_raw:
                        raise ValueError(
                            f"CLS metric point '{cls_metric_point}' not captured; "
                            f"available points: {list(feats_before_raw.keys())}"
                        )
                    cls_before = feats_before_raw[cls_metric_key]
                    cls_after = feats_after_raw[cls_metric_key]
                    cls_before_norm = _l2_normalize(cls_before, dim=-1)
                    cls_after_norm = _l2_normalize(cls_after, dim=-1)
                    cls_diff = cls_before_norm - cls_after_norm
                    metric_buffer["cls"]["norm_l2"].extend(
                        cls_diff.norm(p=2, dim=-1).detach().cpu().tolist()
                    )
                    metric_buffer["cls"]["norm_l1"].extend(
                        cls_diff.norm(p=1, dim=-1).detach().cpu().tolist()
                    )
                    metric_buffer["cls"]["norm_l1_diff"].extend(
                        (cls_after_norm.norm(p=1, dim=-1).detach().cpu() - cls_before_norm.norm(p=1, dim=-1).detach().cpu()).tolist()
                    )
                    metric_buffer["cls"]["mse"].extend(
                        (cls_before - cls_after).pow(2).mean(dim=-1).detach().cpu().tolist()
                    )
                    if patch_metrics_available:
                        if before_details is None or after_details is None or "tokens" not in before_details or "tokens" not in after_details:
                            raise ValueError("Token details unavailable; rerun with --extra-token-metrics disabled.")
                        patch_before = before_details["tokens"]
                        patch_after = after_details["tokens"]
                        if patch_before.shape[1] != patch_after.shape[1]:
                            if not patch_warning_printed:
                                print(
                                    "[extra] Skipping patch token metrics because token counts differ "
                                    f"(before={patch_before.shape[1]}, after={patch_after.shape[1]})."
                                )
                                patch_warning_printed = True
                            patch_metrics_available = False
                        else:
                            patch_cos = F.cosine_similarity(patch_before, patch_after, dim=-1).mean(dim=-1)
                            metric_buffer["patch"]["cos"].extend(patch_cos.detach().cpu().tolist())
                            patch_before_norm = _l2_normalize(patch_before, dim=-1)
                            patch_after_norm = _l2_normalize(patch_after, dim=-1)
                            patch_diff = patch_before_norm - patch_after_norm
                            metric_buffer["patch"]["norm_l2"].extend(
                                patch_diff.norm(p=2, dim=-1).mean(dim=-1).detach().cpu().tolist()
                            )
                            metric_buffer["patch"]["norm_l1"].extend(
                                patch_diff.norm(p=1, dim=-1).mean(dim=-1).detach().cpu().tolist()
                            )
                            metric_buffer["patch"]["norm_l1_diff"].extend(
                                (patch_after_norm.norm(p=1, dim=-1).mean(dim=-1).detach().cpu() - patch_before_norm.norm(p=1, dim=-1).mean(dim=-1).detach().cpu()).tolist()
                            )
                            metric_buffer["patch"]["mse"].extend(
                                (patch_before - patch_after).pow(2).mean(dim=(-1, -2)).detach().cpu().tolist()
                            )
                    sim_orig = _pairwise_cosine_matrix(cls_before)
                    sim_pruned = _pairwise_cosine_matrix(cls_after)
                    tri_i, tri_j = torch.triu_indices(sim_orig.size(0), sim_orig.size(0), offset=1)
                    sim_orig_ut = sim_orig[tri_i, tri_j]
                    sim_pruned_ut = sim_pruned[tri_i, tri_j]
                    metric_buffer["cls_relation"]["mse"].append(
                        float((sim_pruned_ut - sim_orig_ut).pow(2).mean().item())
                    )
                    metric_buffer["cls_relation"]["linear_cka"].append(
                        float(_linear_cka(cls_before, cls_after).item())
                    )
                    metric_buffer["cls_relation"]["pearson"].append(
                        float(_pearson_corr(sim_orig_ut, sim_pruned_ut).item())
                    )
                    metric_buffer["cls_relation"]["nn_at5"].append(
                        float(_nn_at_k(sim_orig, sim_pruned, k=5).item())
                    )

                total += images.shape[0]
                if max_samples is not None and total >= max_samples:
                    break
    finally:
        close_capture_specs(specs_before)
        close_capture_specs(specs_after)

    return results, token_records, metric_buffer


def summarize(
    results_by_point: Dict[str, List[Dict]],
    per_class_topk: int,
    show_class_summary: bool = True,
):
    summaries = {}
    class_summaries = {}
    block_summary_dict = {}
    for name, entries in results_by_point.items():
        if not entries:
            raise RuntimeError(f"No results collected for capture point '{name}'.")

        cos_values = torch.tensor([item["cosine"] for item in entries], dtype=torch.float64)
        summary = {
            "count": int(cos_values.numel()),
            "mean": float(cos_values.mean().item()),
            "std": float(cos_values.std(unbiased=False).item()),
            "min": float(cos_values.min().item()),
            "max": float(cos_values.max().item()),
        }

        extra_metric_stats = {}
        metric_keys = ("mse", "norm_l2", "norm_l1", "cka", "norm_l1_diff")
        for key in metric_keys:
            filtered = [item[key] for item in entries if key in item and math.isfinite(item[key])]
            if not filtered:
                continue
            tensor = torch.tensor(filtered, dtype=torch.float64)
            stats = {
                "count": int(tensor.numel()),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std(unbiased=False).item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
            }
            extra_metric_stats[key] = stats
            summary[f"{key}_mean"] = stats["mean"]
        if "block" in name:
            block_summary_dict[name] = summary

        per_class = defaultdict(list)
        for item in entries:
            if item["target"] is not None:
                per_class[item["target"]].append(item["cosine"])

        class_summary = []
        if show_class_summary:
            for cls, values in per_class.items():
                tensor = torch.tensor(values, dtype=torch.float64)
                class_summary.append((cls, float(tensor.mean().item()), len(values)))

            class_summary.sort(key=lambda entry: entry[0])
            if class_summary:
                print(f"\n[{name}] Class-wise mean cosine similarities (sorted by class id):")
                for cls, mean_cos, count in class_summary:
                    print(f"  class {cls:<5} mean={mean_cos:.6f} (n={count})")

            class_summary.sort(key=lambda entry: entry[1])
            if per_class_topk > 0 and class_summary:
                print(f"\n[{name}] Lowest {min(per_class_topk, len(class_summary))} class-wise mean cosine similarities:")
                for cls, mean_cos, count in class_summary[:per_class_topk]:
                    print(f"  class {cls:<5} mean={mean_cos:.6f} (n={count})")

        summaries[name] = summary
        class_summaries[name] = class_summary

        if extra_metric_stats:
            print(f"\n[{name}] CLS extra metric summary:")
            print(json.dumps(extra_metric_stats, indent=2))

        print(f"\n[{name}] CLS token cosine similarity summary:")
        print(json.dumps(summary, indent=2))

    summary_block_mean = {}
    for k in summary.keys():
        summary_block_mean[k] = np.mean([v[k] for v in block_summary_dict.values()])
    print("\n[block mean] CLS token cosine similarity summary:")
    print(json.dumps(summary_block_mean, indent=2))
    summaries["block mean"] = summary_block_mean

    return summaries, class_summaries


def summarize_extra_metrics(extra_metrics: Optional[Dict[str, Dict[str, List[float]]]]):
    if not extra_metrics:
        return None
    summary = {}
    for section, metrics in extra_metrics.items():
        section_summary = {}
        for name, values in metrics.items():
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float64)
            stats = {
                "count": len(values),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std(unbiased=False).item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
            }
            section_summary[name] = stats
            print(f"\n[{section}] {name} statistics:")
            print(json.dumps(stats, indent=2))
        if section_summary:
            summary[section] = section_summary
    return summary


def report_pruning_impact(
    results_by_point: Dict[str, List[Dict]],
    token_records: List[Dict],
    reference_point: str,
    bottom_k: int,
    pruning_loc: Optional[List[int]] = None,
):
    if reference_point not in results_by_point:
        print(f"[impact] Reference capture '{reference_point}' not found; available keys: {list(results_by_point)}")
        return
    if not token_records:
        print("[impact] Token details unavailable; rerun with --report-pruning-impact or --dump-token-details.")
        return
    entries = sorted(results_by_point[reference_point], key=lambda item: item["cosine"])
    if not entries:
        print(f"[impact] No cosine entries recorded for '{reference_point}'.")
        return
    bottom_k = max(0, min(bottom_k, len(entries)))
    if bottom_k == 0:
        print(f"[impact] No samples selected for pruning impact analysis (bottom_k={bottom_k}).")
        return
    focus = entries[:bottom_k]
    token_map = {record["index"]: record for record in token_records}

    print(
        f"\n[impact] Inspecting lowest {bottom_k} cosine samples at '{reference_point}' "
        f"(total samples={len(entries)})."
    )
    if pruning_loc:
        stage_labels = [f"stage{idx}:block{blk}" for idx, blk in enumerate(pruning_loc)]
        print("[impact] Pruning stage order: " + ", ".join(stage_labels))

    stage_count = 0
    for record in token_records:
        masks = record.get("kept_masks", [])
        stage_count = max(stage_count, len(masks))
    if stage_count == 0:
        print("[impact] No kept_masks recorded; cannot summarize per-stage pruning.")
        return

    for stage_idx in range(stage_count):
        kept_values: List[Tuple[float, int]] = []
        for entry in focus:
            rec = token_map.get(entry["index"])
            if not rec:
                continue
            masks = rec.get("kept_masks", [])
            if len(masks) <= stage_idx:
                continue
            mask = masks[stage_idx]
            if not mask:
                continue
            total_tokens = len(mask)
            kept_tokens = float(sum(mask))
            kept_values.append((kept_tokens, total_tokens))
        if not kept_values:
            print(f"[impact] Stage {stage_idx}: no mask data available.")
            continue
        total_tokens = kept_values[0][1]
        avg_keep = sum(v[0] for v in kept_values) / len(kept_values)
        min_keep = min(v[0] for v in kept_values)
        max_keep = max(v[0] for v in kept_values)
        ratio = avg_keep / total_tokens if total_tokens > 0 else 0.0
        stage_label = f"block{pruning_loc[stage_idx]}" if pruning_loc and stage_idx < len(pruning_loc) else f"stage{stage_idx}"
        print(
            f"[impact] {stage_label:<12} mean keep {avg_keep:.1f}/{total_tokens} ({ratio*100:.2f}%), "
            f"range [{min_keep:.0f}, {max_keep:.0f}] over {len(kept_values)} samples."
        )

    preview = min(5, len(focus))
    if preview > 0:
        print("[impact] Example samples (index, class, cosine, kept-per-stage):")
        for entry in focus[:preview]:
            rec = token_map.get(entry["index"], {})
            mask_summary = rec.get("kept_masks", [])
            keep_counts = [int(round(sum(mask))) for mask in mask_summary]
            target = entry["target"] if entry["target"] is not None else "-"
            print(
                f"  idx={entry['index']:>4} class={target:>4} cos={entry['cosine']:.4f} "
                f"kept={keep_counts}"
            )


def calc_cls_similarity(model_before, model_after, data_loader, device, output_json):
    model_before.to(device).eval()
    model_after.to(device).eval()

    capture_points = ["norm"]

    extra_blocks = [f"block{idx}" for idx in range(len(model_before.blocks))]
    capture_points = list(dict.fromkeys(capture_points + extra_blocks))

    results, token_records, extra_metrics = cosine_similarity_distribution(
        model_before,
        model_after,
        data_loader,
        device,
        capture_points,
        collect_token_details=False,
        extra_metrics=False,
        cls_metric_point="norm",
        patch_metrics_available=False,
    )

    summary, class_summary = summarize(
        results,
        per_class_topk=0,
        show_class_summary=False,
    )
    extra_summary = summarize_extra_metrics(extra_metrics)

    payload = {"summary": summary}
    if extra_summary is not None:
        payload["extra_metrics"] = extra_summary
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



def main():
    args = parse_args()
    device = torch.device(args.device)

    before_dims = infer_structured_mlp_dims(args.before_checkpoint)
    after_dims = infer_structured_mlp_dims(args.after_checkpoint)
    if before_dims:
        print(f"[before] Structured MLP dims: {before_dims}")
    if after_dims and after_dims != before_dims:
        print(f"[after] Structured MLP dims: {after_dims}")
    if before_dims is None and after_dims is None:
        raise RuntimeError("Unable to infer structured MLP dims from either checkpoint.")
    if before_dims is None:
        before_dims = after_dims
    if after_dims is None:
        after_dims = before_dims

    # Force the BEFORE model to keep all tokens so it reflects the unpruned baseline.
    model_before = build_vit_model(args, before_dims, keep_all_tokens=True)
    model_after = build_vit_model(args, after_dims)

    load_model_weights(model_before, args.before_checkpoint, args.before_key, args.before_prefix)
    load_model_weights(model_after, args.after_checkpoint, args.after_key, args.after_prefix)

    model_before.to(device).eval()
    model_after.to(device).eval()

    if args.synthetic_samples > 0:
        loader = iter_synthetic_batches(args.synthetic_samples, args.batch_size, args.input_size)
        data_iter = loader
    else:
        if not args.data_path and args.data_set != "image_folder":
            raise ValueError("--data-path must be specified for the selected dataset.")
        dataset_val, _ = build_dataset(is_train=False, args=args)
        data_iter = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            shuffle=False,
            drop_last=False,
        )

    max_samples = args.max_samples if args.max_samples > 0 else None
    dump_details_path = args.dump_token_details or None
    collect_token_details = bool(dump_details_path or args.report_pruning_impact)
    if collect_token_details:
        print("Token detail capture enabled; recommend using --max-samples to limit file size.")
    capture_points = list(dict.fromkeys(args.capture_points))
    if args.capture_all_blocks:
        extra_blocks = [f"block{idx}" for idx in range(len(model_before.blocks))]
        capture_points = list(dict.fromkeys(capture_points + extra_blocks))

    results, token_records, extra_metrics = cosine_similarity_distribution(
        model_before,
        model_after,
        data_iter,
        device,
        capture_points,
        max_samples=max_samples,
        collect_token_details=collect_token_details,
        extra_metrics=args.extra_token_metrics,
        cls_metric_point=args.cls_metric_point,
    )

    summary, class_summary = summarize(
        results,
        args.per_class_topk,
        show_class_summary=args.show_class_summary,
    )
    extra_summary = None
    if args.extra_token_metrics:
        extra_summary = summarize_extra_metrics(extra_metrics)

    if args.report_pruning_impact:
        report_pruning_impact(
            results,
            token_records,
            args.impact_reference,
            args.impact_bottom_k,
            getattr(model_after, "pruning_loc", None),
        )

    if args.output_json:
        payload = {"summary": summary, "per_class": class_summary, "samples": results}
        if extra_summary is not None:
            payload["extra_metrics"] = extra_summary
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote detailed results to {args.output_json}")

    if dump_details_path:
        with open(dump_details_path, "w", encoding="utf-8") as f:
            json.dump({"samples": token_records}, f)
        print(f"Wrote pruning token details to {dump_details_path}")


if __name__ == "__main__":
    main()
