
#python structured_pruning/train_with_gate_pruning.py --model deit-b --data_set CIFAR --data_path /home/kouki/datasets/cifar100 --nb_classes 100 --input_size 224 --batch_size 128 --epochs 120 --lr 5e-4 --weight_decay 0.05 --drop_path 0.2 --token-rate 0.7 0.49 0.343 --ratio_weight 5.0 --finetune outputs/token/base-rate70/checkpoint-best.pth --seed 42 --gate-enable --gate-target-keep 0.60 --gate-warmup-epochs 20 --gate-prune-interval 5 --gate-l1 5e-4 --gate-min-channels 96 --cls-sim-ref outputs/token/base-rate70/checkpoint-best.pth --cls-sim-weight 0.5 --cls-sim-layers 6 7 8 9 10 11 --cls-sim-feature tokens --output_dir outputs/structured_gate --log_dir logs/structured_gate
#!/usr/bin/env python3
"""
Single-process training entry point that injects channel-wise MLP gates and
gradually prunes them during training.

This script is intentionally scoped to DeiT-style DynamicViT runs on CIFAR-100.
It reuses the same CLI as main.py plus a few gate-specific options.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from main import (
    load_checkpoint,
)
from models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
from timm.data import Mixup
from timm.utils import ModelEma
from utils import NativeScalerWithGradNormCount as NativeScaler


class GateUnit(nn.Module):
    """
    Lightweight gating module attached to each block MLP.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(hidden_dim))
        self.register_buffer("mask", torch.ones(hidden_dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.logits) * self.mask
        return tensor * gate.view(1, 1, -1)

    def l1_penalty(self) -> torch.Tensor:
        gate = torch.sigmoid(self.logits) * self.mask
        return gate.mean()

    def keep_ratio(self) -> float:
        return float(self.mask.sum().item() / self.mask.numel())

    def active_indices(self) -> torch.Tensor:
        return (self.mask > 0.5).nonzero(as_tuple=False).flatten()

    def hard_prune(self, keep_idx: torch.Tensor) -> None:
        new_mask = torch.zeros_like(self.mask)
        if keep_idx.numel() > 0:
            new_mask[keep_idx] = 1.0
        self.mask.copy_(new_mask)
        with torch.no_grad():
            off_idx = (new_mask < 0.5).nonzero(as_tuple=False).flatten()
            if off_idx.numel() > 0:
                self.logits.data[off_idx] = -12.0

    def clone_subset(self, keep_idx: torch.Tensor) -> GateUnit:
        keep_idx = keep_idx.detach()
        new_gate = GateUnit(int(keep_idx.numel()))
        if keep_idx.numel() == 0:
            return new_gate
        with torch.no_grad():
            new_gate.logits.copy_(self.logits.detach()[keep_idx])
        return new_gate.to(self.logits.device, dtype=self.logits.dtype)


class GateController(nn.Module):
    """
    Attaches GateUnit modules to every transformer block and exposes utilities
    for regularization & scheduled pruning.
    """

    def __init__(self, model: nn.Module, min_channels: int = 64):
        super().__init__()
        if not hasattr(model, "blocks"):
            raise ValueError("GateController expects a model with .blocks")
        self.blocks = list(model.blocks)
        self.min_channels = min_channels
        self.gates = nn.ModuleList()
        self.handles = []
        self.initial_dims: List[int] = []
        for block in self.blocks:
            hidden = block.mlp.fc1.weight.shape[0]
            gate = GateUnit(hidden)
            self.gates.append(gate)
            self.initial_dims.append(hidden)
        self._attach_hooks()

    def _attach_hooks(self):
        self._remove_hooks()
        self.handles = []
        for block, gate in zip(self.blocks, self.gates):
            handle = block.mlp.fc1.register_forward_hook(
                lambda _module, _inputs, output, g=gate: g(output)
            )
            self.handles.append(handle)

    def _remove_hooks(self):
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []

    def regularization(self) -> torch.Tensor:
        penalties = [gate.l1_penalty() for gate in self.gates]
        return torch.stack(penalties).mean()

    def current_keep_ratio(self) -> float:
        ratios = []
        for idx, gate in enumerate(self.gates):
            active = float(gate.mask.detach().sum().item())
            base = float(self.initial_dims[idx])
            ratios.append(active / max(base, 1.0))
        return float(sum(ratios) / len(ratios))

    def _select_keep_indices(self, gate: GateUnit, relative: float) -> torch.Tensor:
        active_idx = gate.active_indices()
        if active_idx.numel() == 0:
            return active_idx
        keep_count = max(self.min_channels, int(math.ceil(active_idx.numel() * relative)))
        keep_count = min(keep_count, active_idx.numel())
        with torch.no_grad():
            scores = (
                torch.sigmoid(gate.logits.detach()) * gate.mask.detach()
            )[active_idx]
            topk = torch.topk(scores, keep_count, largest=True).indices
            keep = active_idx[topk]
        return keep.sort().values

    def hard_prune(self, target_keep: float) -> None:
        target_keep = float(max(0.0, min(1.0, target_keep)))
        current_keep = self.current_keep_ratio()
        if target_keep >= current_keep - 1e-5:
            return
        relative = target_keep / max(current_keep, 1e-5)
        for gate, block in zip(self.gates, self.blocks):
            keep = self._select_keep_indices(gate, relative)
            if keep.numel() == 0:
                continue
            gate.hard_prune(keep)
            fc1 = block.mlp.fc1
            fc2 = block.mlp.fc2
            drop_idx = (gate.mask < 0.5).nonzero(as_tuple=False).flatten()
            if drop_idx.numel() > 0:
                fc1.weight.data[drop_idx] = 0.0
                if fc1.bias is not None:
                    fc1.bias.data[drop_idx] = 0.0
                fc2.weight.data[:, drop_idx] = 0.0

    def export_importance(self) -> List[torch.Tensor]:
        return [torch.sigmoid(g.logits.detach()) * g.mask.detach() for g in self.gates]

    def forward(self, x):  # pragma: no cover - unused, Module compatibility only
        return x

    def _apply_physical_prune(self, idx: int, block, gate: GateUnit, keep_idx: torch.Tensor):
        keep_idx = keep_idx.to(block.mlp.fc1.weight.device)
        if keep_idx.numel() == 0:
            raise RuntimeError("Cannot remove all channels during physical pruning.")
        fc1 = block.mlp.fc1
        fc2 = block.mlp.fc2
        device = fc1.weight.device
        dtype = fc1.weight.dtype
        new_hidden = int(keep_idx.numel())
        new_fc1 = nn.Linear(
            fc1.in_features,
            new_hidden,
            bias=fc1.bias is not None,
        ).to(device=device, dtype=dtype)
        new_fc2 = nn.Linear(
            new_hidden,
            fc2.out_features,
            bias=fc2.bias is not None,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_fc1.weight.copy_(fc1.weight.detach()[keep_idx])
            if fc1.bias is not None:
                new_fc1.bias.copy_(fc1.bias.detach()[keep_idx])
            new_fc2.weight.copy_(fc2.weight.detach()[:, keep_idx])
            if fc2.bias is not None:
                new_fc2.bias.copy_(fc2.bias.detach())
        block.mlp.fc1 = new_fc1
        block.mlp.fc2 = new_fc2
        new_gate = gate.clone_subset(keep_idx)
        self.gates[idx] = new_gate

    def physical_prune(self, target_keep: float) -> None:
        target_keep = float(max(0.0, min(1.0, target_keep)))
        current_keep = self.current_keep_ratio()
        if target_keep >= current_keep - 1e-5:
            target_keep = current_keep
        relative = target_keep / max(current_keep, 1e-5)
        for idx, (gate, block) in enumerate(zip(self.gates, self.blocks)):
            keep = self._select_keep_indices(gate, relative)
            if keep.numel() == 0:
                continue
            self._apply_physical_prune(idx, block, gate, keep)
        self._attach_hooks()

    def teardown(self):
        self._remove_hooks()

    def describe_structure(self) -> List[str]:
        """
        Return human-readable per-block active channel counts.
        """
        summary = []
        for idx, gate in enumerate(self.gates):
            total = int(self.initial_dims[idx])
            active = int(gate.mask.detach().sum().item())
            ratio = active / max(1, total)
            summary.append(f"Block {idx:02d}: {active}/{total} channels ({ratio:.2%})")
        return summary


class GradientPruner:
    """
    Aggregate gradient importance per MLP channel and prune once at the end.
    """

    def __init__(self, model: nn.Module, min_channels: int = 64):
        if not hasattr(model, "blocks"):
            raise ValueError("GradientPruner expects a model with .blocks")
        self.blocks = list(model.blocks)
        self.min_channels = int(min_channels)
        self.device = next(model.parameters()).device
        self.scores: List[torch.Tensor] = []
        self.original_dims: List[int] = []
        for block in self.blocks:
            hidden = block.mlp.fc1.weight.shape[0]
            self.original_dims.append(hidden)
            self.scores.append(torch.zeros(hidden, device=self.device))

    # 構造化枝刈り重要度計算
    def accumulate(self):
        for idx, block in enumerate(self.blocks):
            fc1 = block.mlp.fc1
            if fc1.weight.grad is None:
                continue
            grad_1 = fc1.weight.grad.detach()
            weight_1 = fc1.weight.detach()
            score = torch.sum(torch.abs(grad_1 * weight_1), dim=1)

            fc2 = block.mlp.fc2
            if fc2.weight.grad is None:
                continue
            grad_2 = fc2.weight.grad.detach()
            weight_2 = fc2.weight.detach()
            score += torch.sum(torch.abs(grad_2 * weight_2), dim=0)

            self.scores[idx][: score.shape[0]] += score

    def _apply_physical_prune(self, block, keep_idx: torch.Tensor):
        fc1 = block.mlp.fc1
        fc2 = block.mlp.fc2
        device = fc1.weight.device
        dtype = fc1.weight.dtype
        new_hidden = int(keep_idx.numel())
        new_fc1 = nn.Linear(
            fc1.in_features,
            new_hidden,
            bias=fc1.bias is not None,
        ).to(device=device, dtype=dtype)
        new_fc2 = nn.Linear(
            new_hidden,
            fc2.out_features,
            bias=fc2.bias is not None,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_fc1.weight.copy_(fc1.weight.detach()[keep_idx])
            if fc1.bias is not None:
                new_fc1.bias.copy_(fc1.bias.detach()[keep_idx])
            new_fc2.weight.copy_(fc2.weight.detach()[:, keep_idx])
            if fc2.bias is not None:
                new_fc2.bias.copy_(fc2.bias.detach())
        block.mlp.fc1 = new_fc1
        block.mlp.fc2 = new_fc2

    def prune(self, target_keep: float, prune_layers: list[int] = None):
        target_keep = float(max(0.0, min(1.0, target_keep)))
        for idx, block in enumerate(self.blocks):
            # prune_layersに含まれるidの層の枝刈り
            # prune_layersがNoneなら全ての層を刈る
            if (prune_layers is not None) and (idx not in prune_layers):
                continue
            scores = self.scores[idx]
            channels = scores.numel()
            keep = max(self.min_channels, int(math.ceil(channels * target_keep)))
            keep = min(max(1, keep), channels)
            if keep >= channels:
                continue
            topk = torch.topk(scores, keep, largest=True).indices
            keep_idx = torch.sort(topk).values
            self._apply_physical_prune(block, keep_idx)
            self.scores[idx] = scores[keep_idx]

    def describe_structure(self) -> List[str]:
        summary = []
        for idx, block in enumerate(self.blocks):
            total = self.original_dims[idx]
            active = block.mlp.fc1.out_features
            ratio = active / max(1, total)
            summary.append(f"Block {idx:02d}: {active}/{total} channels ({ratio:.2%})")
        return summary


class ClsSimilarityRegularizer:
    """
    Encourages cosine similarity between intermediate representations of the student
    and a frozen reference checkpoint. The regularization target can be the CLS token
    (previous behaviour) or all spatial tokens.
    """

    def __init__(
        self,
        student: nn.Module,
        reference: nn.Module,
        layers: Optional[Sequence[int]],
        weight: float,
        device: torch.device,
        feature_type: str = "cls",
        loss_type: str = "cosine",
    ):
        self.reference = reference.to(device)
        self.reference.eval()
        for param in self.reference.parameters():
            param.requires_grad_(False)
        self.layers: Set[int] = (
            set(layers) if layers is not None and len(layers) > 0 else set(range(len(student.blocks)))
        )
        self.weight = float(weight)
        feature_type = (feature_type or "cls").lower()
        if feature_type not in {"cls", "tokens"}:
            raise ValueError(f"Unsupported feature_type '{feature_type}', expected 'cls' or 'tokens'.")
        self.feature_type = feature_type
        loss_type = (loss_type or "cosine").lower()
        if loss_type not in {"cosine", "l2", "cosine_l2"}:
            raise ValueError(f"Unsupported loss_type '{loss_type}', expected 'cosine', 'l2', or 'cosine_l2'.")
        self.loss_type = loss_type
        self.student_cache: dict[int, torch.Tensor] = {}
        self.ref_cache: dict[int, torch.Tensor] = {}
        self.student_handles = []
        self.ref_handles = []
        self.device = device
        for idx, block in enumerate(student.blocks):
            if idx not in self.layers:
                continue
            self.student_handles.append(
                block.register_forward_hook(self._make_hook(self.student_cache, idx))
            )
        for idx, block in enumerate(self.reference.blocks):
            if idx not in self.layers:
                continue
            self.ref_handles.append(
                block.register_forward_hook(self._make_hook(self.ref_cache, idx))
            )

    def _make_hook(self, cache, idx: int):
        def hook(_module, _inputs, output):
            cache[idx] = self._select_feature(output)
            return output
        return hook

    def _select_feature(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.feature_type == "cls":
            return tensor[:, 0, :]
        tokens = tensor[:, 1:, :]
        if tokens.numel() == 0:
            # Degenerates to CLS if no spatial tokens are available.
            return tensor[:, 0, :]
        return tokens

    def _align_features(
        self, student_feat: torch.Tensor, ref_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_type == "tokens":
            if student_feat.dim() != 3 or ref_feat.dim() != 3:
                raise ValueError("Token similarity expects 3D tensors (batch, tokens, dim).")
            min_tokens = min(student_feat.shape[1], ref_feat.shape[1])
            if min_tokens == 0:
                raise ValueError("Token similarity requires at least one spatial token.")
            student_feat = student_feat[:, :min_tokens].contiguous()
            ref_feat = ref_feat[:, :min_tokens].contiguous()
        return student_feat, ref_feat

    def prepare(self):
        self.student_cache.clear()
        self.ref_cache.clear()

    def compute_loss(self) -> torch.Tensor:
        if not self.layers or self.weight <= 0:
            return torch.zeros(1, device=self.device).squeeze(0)
        penalties = []
        for idx in self.layers:
            student_feat = self.student_cache.get(idx)
            ref_feat = self.ref_cache.get(idx)
            if student_feat is None or ref_feat is None:
                continue
            penalties.append(self._layer_penalty(student_feat, ref_feat))
        if not penalties:
            return torch.zeros(1, device=self.device).squeeze(0)
        stacked = torch.stack(penalties).mean()
        return self.weight * stacked

    def _layer_penalty(self, student_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        student_feat, ref_feat = self._align_features(student_feat, ref_feat)
        if self.loss_type == "l2":
            diff = student_feat - ref_feat
            return diff.pow(2).mean()
        student_norm = F.normalize(student_feat, dim=-1)
        ref_norm = F.normalize(ref_feat, dim=-1)
        cos = (student_norm * ref_norm).sum(dim=-1)
        # torch.absは必要ないと思われる
        cos_loss = torch.abs(1.0 - cos).mean()
        if self.loss_type == "cosine":
            return cos_loss
        diff = student_feat - ref_feat
        mag_loss = diff.pow(2).mean()
        return mag_loss + cos_loss

    def run_reference(self, samples: torch.Tensor):
        with torch.no_grad():
            self.reference(samples)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, warmup_steps=-1):
    return utils.cosine_scheduler(
        base_value,
        final_value,
        epochs,
        niter_per_ep,
        warmup_epochs=warmup_epochs,
        warmup_steps=warmup_steps,
    )


def load_teacher(args, device):
    teacher = VisionTransformerTeacher(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=args.nb_classes,
    )
    ckpt_path = args.teacher_path or args.finetune
    if ckpt_path and os.path.isfile(ckpt_path):
        checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        utils.load_state_dict(teacher, state)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _resolve_token_keep(args, token_ratio_override=None):
    if token_ratio_override is not None:
        return [float(v) for v in token_ratio_override]
    if args.token_rate is not None:
        if len(args.token_rate) != 3:
            raise ValueError(f"--token-rate expects 3 values, got {len(args.token_rate)}.")
        return [float(v) for v in args.token_rate]
    base_rate = float(args.base_rate)
    return [base_rate, base_rate ** 2, base_rate ** 3]


def build_student_model(args, mlp_hidden_dims=None, token_ratio_override=None):
    keep = _resolve_token_keep(args, token_ratio_override=token_ratio_override)
    model = VisionTransformerDiffPruning(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        pruning_loc=[3, 6, 9],
        token_ratio=keep,
        distill=True,
        drop_path_rate=args.drop_path,
        mlp_hidden_dims=mlp_hidden_dims,
        num_classes=args.nb_classes,
    )
    return model


def train_one_epoch_with_gates(
    model: nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScaler,
    max_norm: float,
    model_ema: Optional[ModelEma],
    mixup_fn: Optional[Mixup],
    lr_schedule,
    wd_schedule,
    num_training_steps_per_epoch: int,
    cls_regularizer: Optional[ClsSimilarityRegularizer],
    grad_tracker: Optional[GradientPruner] = None,
    only_grad_accumulate: bool = False,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f"Epoch [{epoch}]"
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        step = data_iter_step
        if step >= num_training_steps_per_epoch:
            break
        it = epoch * num_training_steps_per_epoch + step
        if lr_schedule is not None:
            for param_group in optimizer.param_groups:
                base_lr = lr_schedule[min(it, len(lr_schedule) - 1)]
                lr_scale = param_group.get("lr_scale", 1.0)
                fix_step = param_group.get("fix_step", 0)
                if epoch < fix_step:
                    param_group["lr"] = 0.0
                else:
                    param_group["lr"] = base_lr * lr_scale
                if wd_schedule is not None and param_group.get("weight_decay", 0) > 0:
                    param_group["weight_decay"] = wd_schedule[min(it, len(wd_schedule) - 1)]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if cls_regularizer is not None:
            cls_regularizer.prepare()

        token_keep_values: Optional[List[float]] = None
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(samples)
            loss, loss_part = criterion(samples, outputs, targets)

            # Track how many tokens survive each pruning stage during training.
            if isinstance(outputs, (list, tuple)) and outputs:
                stage_probs = outputs[-1]
                if isinstance(stage_probs, (list, tuple)):
                    token_keep_values = []
                    for stage in stage_probs:
                        if torch.is_tensor(stage):
                            keep_val = stage.detach().float().mean().item()
                            token_keep_values.append(keep_val)
                    if not token_keep_values:
                        token_keep_values = None
            if cls_regularizer is not None and cls_regularizer.weight > 0:
                cls_regularizer.run_reference(samples)
                loss_cls = cls_regularizer.compute_loss()
                loss = loss + loss_cls
                cos_loss_val = loss_cls.detach().item()
            else:
                cos_loss_val = 0.0

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Loss is {loss_value}, stopping training")
        loss.backward()

        if grad_tracker is not None:
            grad_tracker.accumulate()
        if not only_grad_accumulate:
            optimizer.step()
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(cls_cos_loss=cos_loss_val)
        if loss_part and len(loss_part) > 1 and loss_part[1] is not None:
            metric_logger.update(token_ratio_loss=loss_part[1].item())
        if token_keep_values:
            for idx, keep_val in enumerate(token_keep_values):
                metric_logger.update(**{f"token_keep_s{idx}": keep_val})
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
