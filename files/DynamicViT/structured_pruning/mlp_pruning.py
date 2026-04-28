"""
Helpers to prune VisionTransformerDiffPruning MLP layers by removing low-norm hidden channels.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PruningStats:
    keep_indices: List[torch.Tensor]
    original_dims: List[int]
    pruned_dims: List[int]

    def as_dict(self) -> Dict[str, List[int]]:
        return {
            "original_dims": self.original_dims,
            "pruned_dims": self.pruned_dims,
        }


def _extract_logits(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unsupported model output type {type(output)} when computing importance.")


def _fetch_images(batch):
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError(f"Unable to extract image tensor from batch of type {type(batch)}.")


def _channel_norms(fc1: nn.Linear, fc2: nn.Linear, norm_type: str = "l2") -> torch.Tensor:
    """Compute channel importance shared between fc1 rows and fc2 columns."""
    with torch.no_grad():
        w1 = fc1.weight.detach()
        w2 = fc2.weight.detach()
        bias = fc1.bias.detach() if fc1.bias is not None else None

        if norm_type == "l1":
            norms = w1.abs().sum(dim=1)
            norms = norms + w2.abs().sum(dim=0)
            if bias is not None:
                norms = norms + bias.abs()
            return norms

        if norm_type != "l2":
            raise ValueError(f"Unsupported norm_type '{norm_type}', expected 'l1' or 'l2'.")

        norms = w1.pow(2).sum(dim=1)
        norms = norms + w2.pow(2).sum(dim=0)
        if bias is not None:
            norms = norms + bias.pow(2)
        return norms.sqrt()


def _prune_linear(layer: nn.Linear, idx, dim: int) -> nn.Linear:
    keep = idx.to(layer.weight.device)
    in_features = layer.in_features if dim == 0 else len(keep)
    out_features = len(keep) if dim == 0 else layer.out_features
    new_linear = nn.Linear(in_features, out_features, bias=layer.bias is not None)
    new_linear = new_linear.to(device=layer.weight.device, dtype=layer.weight.dtype)

    with torch.no_grad():
        if dim == 0:
            new_linear.weight.copy_(layer.weight[keep])
            if layer.bias is not None:
                new_linear.bias.copy_(layer.bias[keep])
        else:
            new_linear.weight.copy_(layer.weight[:, keep])
            if layer.bias is not None:
                new_linear.bias.copy_(layer.bias)
    return new_linear


def compute_distill_channel_importance(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    max_batches: Optional[int] = 64,
    teacher_model: Optional[nn.Module] = None,
) -> List[torch.Tensor]:
    """
    Use a KL divergence objective to retain the output distribution of the reference model.

    The student model shares weights with the teacher and receives multiplicative gating
    parameters after fc1. We accumulate the average gate gradient magnitude as an importance
    estimate per hidden channel.
    """
    if data_loader is None:
        raise ValueError("Data loader is required to compute distillation-based importance.")

    teacher = teacher_model
    if teacher is None:
        teacher = deepcopy(model)
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    student = model
    student.eval()
    for param in student.parameters():
        param.requires_grad_(False)

    gates: List[torch.Tensor] = []
    handles = []

    for block in student.blocks:
        hidden = block.mlp.fc1.weight.shape[0]
        gate = torch.ones(hidden, device=device, requires_grad=True)
        gates.append(gate)

        def _hook(_module, _inputs, output, gate_tensor=gate):
            return output * gate_tensor.view(1, 1, -1)

        handles.append(block.mlp.fc1.register_forward_hook(_hook))

    importance = [torch.zeros_like(gate) for gate in gates]
    processed = 0
    try:
        for batch in data_loader:
            if max_batches is not None and max_batches > 0 and processed >= max_batches:
                break
            images = _fetch_images(batch).to(device, non_blocking=True)
            with torch.no_grad():
                teacher_logits = _extract_logits(teacher(images))
            student_logits = _extract_logits(student(images))
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.log_softmax(teacher_logits, dim=-1),
                reduction="batchmean",
                log_target=True,
            )
            loss.backward()
            processed += 1
            for idx, gate in enumerate(gates):
                grad = gate.grad
                if grad is None:
                    continue
                importance[idx] += grad.detach().abs()
                gate.grad.zero_()
            student.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()

    if processed == 0:
        raise RuntimeError("No calibration batches were processed for importance estimation.")
    return [imp.detach().cpu() / processed for imp in importance]


def prune_block_mlp(
    block: nn.Module,
    keep_ratio: float,
    min_channels: int = 64,
    norm_type: str = "l2",
    importance_scores: Optional[torch.Tensor] = None,
) -> Tuple[int, torch.Tensor]:
    """Prune MLP hidden channels inside a transformer block."""
    mlp = block.mlp
    fc1: nn.Linear = mlp.fc1
    fc2: nn.Linear = mlp.fc2
    orig_hidden = fc1.weight.shape[0]
    target = max(min_channels, int(orig_hidden * keep_ratio))
    target = min(target, orig_hidden)
    if target == orig_hidden:
        return orig_hidden, torch.arange(orig_hidden, device=fc1.weight.device)

    if importance_scores is not None:
        norms = importance_scores.to(fc1.weight.device)
    else:
        norms = _channel_norms(fc1, fc2, norm_type=norm_type)
    keep_idx = torch.topk(norms, target, largest=True).indices
    keep_idx, _ = keep_idx.sort()

    mlp.fc1 = _prune_linear(fc1, keep_idx, dim=0)
    mlp.fc2 = _prune_linear(fc2, keep_idx, dim=1)
    return target, keep_idx


def prune_model_mlp_channels(
    model: nn.Module,
    keep_ratio: float,
    min_channels: int = 64,
    norm_type: str = "l2",
    importance_scores: Optional[Sequence[torch.Tensor]] = None,
) -> PruningStats:
    """Apply channel pruning to every block in the model."""
    original = []
    pruned = []
    keep_indices: List[torch.Tensor] = []
    for idx, block in enumerate(model.blocks):
        hidden = block.mlp.fc1.weight.shape[0]
        original.append(int(hidden))
        block_scores = None
        if importance_scores is not None and idx < len(importance_scores):
            block_scores = importance_scores[idx]
        new_hidden, keep_idx = prune_block_mlp(
            block,
            keep_ratio,
            min_channels,
            norm_type=norm_type,
            importance_scores=block_scores,
        )
        pruned.append(int(new_hidden))
        keep_indices.append(keep_idx.cpu())
    return PruningStats(keep_indices=keep_indices, original_dims=original, pruned_dims=pruned)


def summarize_pruning(stats: PruningStats) -> str:
    deltas = [o - n for o, n in zip(stats.original_dims, stats.pruned_dims)]
    lines = ["MLP structured pruning summary:"]
    for idx, (orig, new, removed) in enumerate(zip(stats.original_dims, stats.pruned_dims, deltas)):
        ratio = new / orig if orig > 0 else 0.0
        lines.append(f"  Block {idx:02d}: {orig} -> {new} (removed {removed}, keep {ratio:.2%})")
    total_orig = sum(stats.original_dims)
    total_new = sum(stats.pruned_dims)
    lines.append(f"Total hidden dims: {total_orig} -> {total_new} (keep {total_new / total_orig:.2%})")
    return "\n".join(lines)
