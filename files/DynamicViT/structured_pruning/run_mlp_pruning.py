#!/usr/bin/env python3
"""
Pipeline for MLP-only structured pruning using channel norms followed by checkpoint export.

Example:
    python structured_pruning/run_mlp_pruning.py \\
        --checkpoint-in pretrained/deit_base_patch16_224-b5f2ef4d.pth \\
        --checkpoint-out outputs/pruned_mlp/checkpoint.pth \\
        --prune-ratio 0.3 --num-classes 100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List, Optional

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets as tv_datasets, transforms as tv_transforms
from torchvision.transforms import InterpolationMode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from main import infer_structured_mlp_dims, load_checkpoint
from models.dyvit import VisionTransformerDiffPruning
from structured_pruning.mlp_pruning import (
    compute_distill_channel_importance,
    prune_model_mlp_channels,
    summarize_pruning,
)
import utils


def parse_args():
    parser = argparse.ArgumentParser("Structured MLP pruning via channel norms")
    parser.add_argument("--checkpoint-in", default="pretrained/deit_base_patch16_224-b5f2ef4d.pth",
                        help="Input checkpoint (pre-pruning).")
    parser.add_argument("--checkpoint-out", required=True, help="Path to save the pruned checkpoint.")
    parser.add_argument("--model", default="deit-b", choices=["deit-b"], help="Model backbone to instantiate.")
    parser.add_argument("--num-classes", type=int, default=100, help="Classifier output dimension.")
    parser.add_argument("--prune-ratio", type=float, default=0.25,
                        help="Fraction of hidden channels to remove in each MLP (0-1).")
    parser.add_argument("--min-channels", type=int, default=96, help="Minimum hidden channels per block.")
    parser.add_argument("--token-rate", type=float, nargs=3, default=None,
                        help="Explicit keep ratio (0-1) for each pruning stage.")
    parser.add_argument("--base-rate", type=float, default=0.7,
                        help="Base keep ratio used when token-rate is not specified.")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Skip saving, only report pruning summary.")
    parser.add_argument("--show-model", action="store_true",
                        help="Print pruned model architecture and parameter count after pruning.")
    parser.add_argument("--importance", type=str, default="l2", choices=["l1", "l2", "kl"],
                        help="Channel ranking metric. Use 'kl' to approximate the teacher output distribution.")
    parser.add_argument("--data-set", default="CIFAR", choices=["CIFAR", "IMNET", "image_folder"],
                        help="Dataset used for KL-based importance calculation.")
    parser.add_argument("--data-path", default=None,
                        help="Dataset root (CIFAR or ImageNet). Required when --importance kl.")
    parser.add_argument("--eval-data-path", default=None,
                        help="Optional evaluation folder when data_set=image_folder.")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Input resolution for calibration images.")
    parser.add_argument("--imagenet-default-mean-and-std", type=utils.str2bool, default=True,
                        help="Use ImageNet mean/std for calibration preprocessing.")
    parser.add_argument("--calib-batch-size", type=int, default=64,
                        help="Batch size during KL-based importance estimation.")
    parser.add_argument("--calib-batches", type=int, default=64,
                        help="Number of calibration batches (<=0 uses the entire dataset).")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers for KL importance.")
    parser.add_argument("--pin-mem", type=utils.str2bool, default=True,
                        help="Pin memory for the calibration DataLoader.")
    parser.add_argument("--teacher-checkpoint", default=None,
                        help="Optional checkpoint used as the teacher when --importance kl.")
    parser.add_argument("--disable-token-pruning", action="store_true",
                        help="Instantiate the student without DynamicViT token pruning heads.")
    return parser.parse_args()


def _calibration_mean_std(args):
    if args.imagenet_default_mean_and_std:
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


def build_calibration_transform(args):
    mean, std = _calibration_mean_std(args)
    t = []
    if args.input_size > 32:
        if args.input_size >= 384:
            t.append(
                tv_transforms.Resize(
                    (args.input_size, args.input_size),
                    interpolation=InterpolationMode.BICUBIC,
                )
            )
        else:
            t.append(tv_transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC))
            t.append(tv_transforms.CenterCrop(args.input_size))
    else:
        t.append(tv_transforms.Resize((args.input_size, args.input_size)))
    t.append(tv_transforms.ToTensor())
    t.append(tv_transforms.Normalize(mean, std))
    return tv_transforms.Compose(t)


def build_calibration_dataset(args):
    transform = build_calibration_transform(args)
    name = args.data_set.lower()
    if name == "cifar":
        if args.data_path is None:
            raise ValueError("--data-path is required for CIFAR calibration.")
        dataset = tv_datasets.CIFAR100(args.data_path, train=False, transform=transform)
    elif name == "imnet":
        if args.data_path is None:
            raise ValueError("--data-path is required for ImageNet calibration.")
        root = os.path.join(args.data_path, "val")
        dataset = tv_datasets.ImageFolder(root, transform=transform)
    elif name == "image_folder":
        root = args.eval_data_path or args.data_path
        if root is None:
            raise ValueError("Specify --data-path or --eval-data-path for image_folder calibration.")
        dataset = tv_datasets.ImageFolder(root, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset '{args.data_set}'.")
    return dataset


def build_calibration_loader(args):
    dataset = build_calibration_dataset(args)
    sampler = SequentialSampler(dataset)
    batch_size = max(1, int(args.calib_batch_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )


def _deit_config(model_name: str) -> dict:
    if model_name == "deit-b":
        return dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.)
    raise ValueError(f"Unsupported model {model_name} for structured pruning.")


def build_model(args, mlp_hidden_dims: Optional[List[int]]):
    cfg = _deit_config(args.model)
    pruning_loc = [] if args.disable_token_pruning else [3, 6, 9]
    if args.token_rate is not None:
        keep = list(args.token_rate)
    else:
        base = args.base_rate
        keep = [base, base ** 2, base ** 3]
    if args.disable_token_pruning:
        keep = [1.0, 1.0, 1.0]
        print("[structured pruning] Token pruning disabled; removing only MLP channels.")

    model = VisionTransformerDiffPruning(
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        qkv_bias=True,
        pruning_loc=pruning_loc,
        token_ratio=keep,
        distill=True,
        drop_path_rate=args.drop_path,
        mlp_hidden_dims=mlp_hidden_dims,
        num_classes=args.num_classes,
    )
    return model


def load_weights(model, path: str):
    checkpoint = load_checkpoint(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in state and key in model_state and state[key].shape != model_state[key].shape:
            print(f"[Pruning] Removing incompatible key {key} from checkpoint {path}.")
            del state[key]
    utils.load_state_dict(model, state)


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = os.path.dirname(args.checkpoint_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    utils.log_cli_command(Path(args.checkpoint_out).parent / "run_command.txt")

    structured_dims = infer_structured_mlp_dims(args.checkpoint_in)
    if structured_dims is not None:
        print(f"Loaded structured MLP dimensions from checkpoint ({len(structured_dims)} layers).")

    model = build_model(args, structured_dims)
    load_weights(model, args.checkpoint_in)
    model.to(device)
    model.eval()

    importance_scores = None
    if args.importance == "kl":
        print("Computing KL-based channel importance...")
        calib_loader = build_calibration_loader(args)
        teacher_model = None
        if args.teacher_checkpoint:
            teacher_model = build_model(args, structured_dims)
            load_weights(teacher_model, args.teacher_checkpoint)
            teacher_model.to(device)
            teacher_model.eval()
        importance_scores = compute_distill_channel_importance(
            model,
            data_loader=calib_loader,
            device=device,
            max_batches=args.calib_batches,
            teacher_model=teacher_model,
        )
        print("Finished computing KL importance.")

    prune_ratio = max(0.0, min(1.0, args.prune_ratio))
    keep_ratio = 1.0 - prune_ratio
    norm_type = args.importance if args.importance in ("l1", "l2") else "l2"
    stats = prune_model_mlp_channels(
        model,
        keep_ratio=keep_ratio,
        min_channels=args.min_channels,
        norm_type=norm_type,
        importance_scores=importance_scores,
    )
    model.mlp_hidden_dims = stats.pruned_dims
    print(summarize_pruning(stats))

    if args.show_model:
        print("\nPruned model architecture:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters after pruning: {total_params}")

    if args.dry_run:
        print("Dry run enabled; skipping checkpoint export.")
        return

    to_save = {
        "model": model.state_dict(),
        "structured_mlp_hidden_dims": stats.pruned_dims,
        "pruning_stats": stats.as_dict(),
        "origin_checkpoint": args.checkpoint_in,
        "keep_ratio": keep_ratio,
        "prune_ratio": prune_ratio,
        "min_channels": args.min_channels,
    }
    torch.save(to_save, args.checkpoint_out)
    print(f"Saved pruned checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
