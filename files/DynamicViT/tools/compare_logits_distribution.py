#!/usr/bin/env python3
"""
Compare the final output distributions of two checkpoints (e.g., before vs. after pruning).

Example:
    python tools/compare_logits_distribution.py \\
        --before-checkpoint outputs/token/base-rate70/checkpoint-best.pth \\
        --after-checkpoint outputs/structured_gate/checkpoint-best.pth \\
        --model deit-b \\
        --data_set CIFAR --data_path /home/kouki/datasets/cifar100 \\
        --nb_classes 100 --batch_size 128 --base_rate 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset
from main import infer_structured_mlp_dims, load_checkpoint
from models.dyvit import VisionTransformerDiffPruning
import utils


MODEL_SPECS: Dict[str, Dict[str, int]] = {
    "deit-s": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "deit-b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
}
PRUNING_LOC = [3, 6, 9]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Compare logits/probability distributions of two checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--before-checkpoint", required=True, help="Reference checkpoint (e.g., before pruning).")
    parser.add_argument("--after-checkpoint", required=True, help="Checkpoint to compare against (e.g., after pruning).")
    parser.add_argument("--model", default="deit-b", choices=sorted(MODEL_SPECS), help="Backbone to instantiate.")
    parser.add_argument("--data_set", default="CIFAR", choices=["CIFAR", "IMNET", "image_folder"],
                        help="Dataset kind passed to datasets.build_dataset.")
    parser.add_argument("--data_path", required=True,
                        help="Dataset root (train split for ImageNet-style, CIFAR root otherwise).")
    parser.add_argument("--eval_data_path", default=None, help="Optional eval folder when data_set=image_folder.")
    parser.add_argument("--nb_classes", type=int, default=100, help="Classifier dimension.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_mem", type=utils.str2bool, default=True)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--imagenet_default_mean_and_std", type=utils.str2bool, default=True)
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument("--token-rate", type=float, nargs=3, default=None,
                        help="Explicit keep ratio for each pruning stage. Falls back to base_rate powers when omitted.")
    parser.add_argument("--base_rate", type=float, default=0.7, help="Base keep ratio to seed token-rate.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional cap on validation batches (debug).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of validation samples to inspect.")
    parser.add_argument("--save-summary", default=None,
                        help="Optional JSON file to store the aggregated statistics.")
    parser.add_argument("--save-details", default=None,
                        help="Optional .pt file containing per-sample divergence metrics.")
    return parser.parse_args()


def _build_model(model_name: str, checkpoint_path: str, args: argparse.Namespace) -> VisionTransformerDiffPruning:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Model '{model_name}' is not supported. Available: {list(MODEL_SPECS)}")
    config = MODEL_SPECS[model_name]
    if args.token_rate is not None:
        keep = list(args.token_rate)
        if len(keep) != len(PRUNING_LOC):
            raise ValueError(f"--token-rate expects {len(PRUNING_LOC)} values.")
    else:
        base = args.base_rate
        keep = [base, base ** 2, base ** 3]

    structured_dims: Optional[List[int]] = infer_structured_mlp_dims(checkpoint_path)
    if structured_dims:
        print(f"[compare] detected structured MLP dims ({len(structured_dims)}) for {checkpoint_path}.")

    model = VisionTransformerDiffPruning(
        patch_size=16,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=4,
        qkv_bias=True,
        pruning_loc=PRUNING_LOC,
        token_ratio=keep,
        distill=True,
        mlp_hidden_dims=structured_dims,
        num_classes=args.nb_classes,
    )
    return model


def _load_weights(model: torch.nn.Module, path: str) -> None:
    checkpoint = load_checkpoint(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print(f"[compare] {path}: missing keys {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"[compare] {path}: unexpected keys {missing.unexpected_keys}")


def _extract_logits(output: torch.Tensor | Sequence[torch.Tensor]):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        return output[0]
    raise TypeError(f"Unsupported model output type {type(output)} when gathering logits.")


def _summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks: Sequence[int]) -> Dict[int, float]:
    valid = [k for k in ks if k <= logits.size(1)]
    if not valid:
        return {k: 0.0 for k in ks}
    max_k = max(valid)
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.unsqueeze(0).expand_as(pred))
    res: Dict[int, float] = {}
    for k in ks:
        if k > logits.size(1):
            res[k] = 0.0
            continue
        res[k] = float(correct[:k].reshape(-1).float().sum().item())
    return res


def main():
    args = _parse_args()
    log_targets = []
    if args.save_summary:
        log_targets.append(Path(args.save_summary).parent / "run_command.txt")
    if args.save_details:
        log_targets.append(Path(args.save_details).parent / "run_command.txt")
    if not log_targets:
        log_targets.append(Path.cwd() / "compare_logits_run_command.txt")
    seen = set()
    for target in log_targets:
        resolved = Path(target)
        key = resolved.resolve()
        if key in seen:
            continue
        seen.add(key)
        utils.log_cli_command(resolved)

    dataset_val, _ = build_dataset(is_train=False, args=args)
    data_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model_before = _build_model(args.model, args.before_checkpoint, args)
    model_after = _build_model(args.model, args.after_checkpoint, args)
    _load_weights(model_before, args.before_checkpoint)
    _load_weights(model_after, args.after_checkpoint)

    device = torch.device(args.device)
    model_before.to(device).eval()
    model_after.to(device).eval()

    metrics: Dict[str, List[float]] = {
        "kl_before_after": [],
        "kl_after_before": [],
        "js_divergence": [],
        "prob_l1": [],
        "prob_l2": [],
        "logit_mse": [],
        "prob_cosine": [],
    }
    details: List[Dict] = []
    total_samples = 0
    agreement_top1 = 0
    acc_before_top1 = 0
    acc_after_top1 = 0
    acc_before_top5 = 0
    acc_after_top5 = 0
    targets_available = args.data_set.lower() != "image_folder" or getattr(dataset_val, "classes", None) is not None

    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(data_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            if args.max_samples is not None and total_samples >= args.max_samples:
                break

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if args.max_samples is not None:
                remaining = args.max_samples - total_samples
                if remaining <= 0:
                    break
                if samples.shape[0] > remaining:
                    samples = samples[:remaining]
                    targets = targets[:remaining]

            logits_before = _extract_logits(model_before(samples))
            logits_after = _extract_logits(model_after(samples))

            log_prob_before = F.log_softmax(logits_before, dim=-1)
            log_prob_after = F.log_softmax(logits_after, dim=-1)
            prob_before = log_prob_before.exp()
            prob_after = log_prob_after.exp()

            kl_before_after = torch.sum(prob_before * (log_prob_before - log_prob_after), dim=-1)
            kl_after_before = torch.sum(prob_after * (log_prob_after - log_prob_before), dim=-1)
            mean_prob = 0.5 * (prob_before + prob_after)
            log_mean = torch.log(mean_prob.clamp_min(1e-12))
            js = 0.5 * (
                torch.sum(prob_before * (log_prob_before - log_mean), dim=-1)
                + torch.sum(prob_after * (log_prob_after - log_mean), dim=-1)
            )
            prob_delta = prob_before - prob_after
            logit_delta = logits_before - logits_after
            prob_l1 = prob_delta.abs().sum(dim=-1)
            prob_l2 = prob_delta.pow(2).sum(dim=-1).sqrt()
            logit_mse = logit_delta.pow(2).mean(dim=-1)
            prob_cosine = F.cosine_similarity(prob_before, prob_after, dim=-1)

            for name, tensor in [
                ("kl_before_after", kl_before_after),
                ("kl_after_before", kl_after_before),
                ("js_divergence", js),
                ("prob_l1", prob_l1),
                ("prob_l2", prob_l2),
                ("logit_mse", logit_mse),
                ("prob_cosine", prob_cosine),
            ]:
                metrics[name].extend(tensor.detach().cpu().tolist())

            preds_before = logits_before.argmax(dim=-1)
            preds_after = logits_after.argmax(dim=-1)
            agreement_top1 += int((preds_before == preds_after).sum().item())

            if targets_available:
                topk_counts = _topk_correct(logits_before, targets, ks=(1, 5))
                acc_before_top1 += topk_counts.get(1, 0.0)
                acc_before_top5 += topk_counts.get(5, 0.0)
                topk_counts = _topk_correct(logits_after, targets, ks=(1, 5))
                acc_after_top1 += topk_counts.get(1, 0.0)
                acc_after_top5 += topk_counts.get(5, 0.0)

            if args.save_details:
                base_index = total_samples
                batch_size = samples.shape[0]
                target_list = targets.detach().cpu().tolist()
                for idx in range(batch_size):
                    entry = {
                        "index": base_index + idx,
                        "target": int(target_list[idx]),
                        "kl_before_after": float(kl_before_after[idx].item()),
                        "kl_after_before": float(kl_after_before[idx].item()),
                        "js_divergence": float(js[idx].item()),
                        "prob_l1": float(prob_l1[idx].item()),
                        "prob_l2": float(prob_l2[idx].item()),
                        "logit_mse": float(logit_mse[idx].item()),
                        "prob_cosine": float(prob_cosine[idx].item()),
                        "top1_before": int(preds_before[idx].item()),
                        "top1_after": int(preds_after[idx].item()),
                        "top1_match": bool(preds_before[idx].item() == preds_after[idx].item()),
                    }
                    details.append(entry)

            total_samples += samples.shape[0]
            if batch_idx % 20 == 0:
                print(f"[compare] processed batch {batch_idx}, total samples {total_samples}")

    if total_samples == 0:
        raise RuntimeError("No samples were processed; check dataset and arguments.")

    summary = {name: _summarize(values) for name, values in metrics.items()}
    summary["samples"] = int(total_samples)
    summary["top1_agreement"] = float(agreement_top1) / total_samples
    if targets_available:
        summary["before_accuracy_top1"] = float(acc_before_top1) / total_samples
        summary["after_accuracy_top1"] = float(acc_after_top1) / total_samples
        summary["before_accuracy_top5"] = float(acc_before_top5) / total_samples
        summary["after_accuracy_top5"] = float(acc_after_top5) / total_samples

    print("\n[compare] Logit distribution summary")
    for name, stats in summary.items():
        if isinstance(stats, dict):
            print(f"  {name}: mean={stats['mean']:.6f} std={stats['std']:.6f} "
                  f"p50={stats['p50']:.6f} p90={stats['p90']:.6f} p99={stats['p99']:.6f}")
        else:
            print(f"  {name}: {stats}")

    if args.save_summary:
        summary_path = Path(args.save_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[compare] Saved summary to {summary_path}")

    if args.save_details:
        details_path = Path(args.save_details)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"details": details, "meta": {
            "before": args.before_checkpoint,
            "after": args.after_checkpoint,
            "model": args.model,
            "data_set": args.data_set,
            "samples": total_samples,
        }}, details_path)
        print(f"[compare] Saved per-sample metrics to {details_path}")


if __name__ == "__main__":
    main()
