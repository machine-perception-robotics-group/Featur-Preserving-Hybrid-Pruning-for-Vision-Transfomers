#!/usr/bin/env python3
"""
Utility script to dump logits (and labels) for a checkpoint on a validation set.

Example:
    python tools/dump_logits.py \
        --checkpoint outputs/token/base-rate70/checkpoint-best.pth \
        --model deit-b \
        --data_set CIFAR \
        --data_path /home/kouki/datasets/cifar100 \
        --batch_size 128 \
        --output logs/base-rate70_logits.pt
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from main import infer_structured_mlp_dims, load_checkpoint
from models.dyvit import VisionTransformerDiffPruning
import utils


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Dump logits for a DeiT checkpoint", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--model", default="deit-b", choices=["deit-b", "deit-s"], help="Backbone to instantiate.")
    parser.add_argument("--data_set", default="CIFAR", choices=["CIFAR", "IMNET", "image_folder"],
                        help="Dataset kind passed to datasets.build_dataset.")
    parser.add_argument("--data_path", required=True, help="Dataset root (train split for ImageNet-style, CIFAR root otherwise).")
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
    parser.add_argument("--output", required=True, help="Path to save logits/labels tensor.")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional cap on validation batches (debug).")
    return parser.parse_args()


def _build_model(args: argparse.Namespace) -> VisionTransformerDiffPruning:
    pruning_loc = [3, 6, 9]
    if args.token_rate is not None:
        keep = list(args.token_rate)
    else:
        base = args.base_rate
        keep = [base, base ** 2, base ** 3]
    structured_dims: Optional[List[int]] = infer_structured_mlp_dims(args.checkpoint)
    if structured_dims:
        print(f"[logits] detected structured MLP dims ({len(structured_dims)} layers).")

    if args.model == "deit-s":
        embed_dim = 384
        depth = 12
        num_heads = 6
    else:
        embed_dim = 768
        depth = 12
        num_heads = 12

    model = VisionTransformerDiffPruning(
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        pruning_loc=pruning_loc,
        token_ratio=keep,
        distill=True,
        mlp_hidden_dims=structured_dims,
        num_classes=args.nb_classes,
    )
    return model


def main():
    args = _parse_args()
    utils.log_cli_command(Path(args.output).parent / "run_command.txt")

    dataset_val, _ = build_dataset(is_train=False, args=args)
    data_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = _build_model(args)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print(f"[logits] missing keys: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"[logits] unexpected keys: {missing.unexpected_keys}")

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(data_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            samples = samples.to(device, non_blocking=True)
            outputs = model(samples)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            logits_list.append(logits.cpu())
            labels_list.append(targets.clone())
            if batch_idx % 50 == 0:
                print(f"[logits] processed batch {batch_idx}")

    if not logits_list:
        raise RuntimeError("No batches were processed; check dataset and arguments.")

    payload = {
        "logits": torch.cat(logits_list, dim=0),
        "labels": torch.cat(labels_list, dim=0),
        "meta": {
            "checkpoint": args.checkpoint,
            "model": args.model,
            "token_rate": args.token_rate,
            "base_rate": args.base_rate,
            "data_set": args.data_set,
            "data_path": args.data_path,
            "input_size": args.input_size,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[logits] Saved logits for {payload['logits'].shape[0]} samples to {output_path}")


if __name__ == "__main__":
    main()
