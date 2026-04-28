"""
Loss helpers that only cover structured MLP pruning.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F


class StructuredPruningDistillLoss(torch.nn.Module):
    """
    Loss used when only the MLP channels are structurally pruned.

    It keeps the original classification loss and knowledge distillation terms
    but deliberately drops any token-ratio regularization to avoid coupling the
    run with token pruning heuristics.
    """

    def __init__(
        self,
        teacher_model: Optional[torch.nn.Module],
        base_criterion: torch.nn.Module,
        clf_weight: float = 1.0,
        distill_weight: float = 0.5,
        token_weight: float = 0.5,
        mse_token: bool = True,
        print_mode: bool = True,
        log_interval: int = 100,
    ) -> None:
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.distill_weight = distill_weight
        self.token_weight = token_weight
        self.mse_token = mse_token
        self.print_mode = print_mode
        self.log_interval = max(1, int(log_interval))

        self._count = 0
        self._cls_loss = 0.0
        self._cls_distill_loss = 0.0
        self._token_distill_loss = 0.0
        print(
            "Structured loss ready:",
            f"clf_weight={clf_weight}",
            f"distill_weight={distill_weight}",
            f"token_weight={token_weight}",
        )

    def _log_progress(self, cls_loss, cls_kl_loss, token_loss) -> None:
        if not self.print_mode:
            return
        self._count += 1
        self._cls_loss += cls_loss
        self._cls_distill_loss += cls_kl_loss
        self._token_distill_loss += token_loss
        if self._count >= self.log_interval:
            print(
                "loss info: cls_loss={:.4f}, cls_kl={:.4f}, token={:.4f}".format(
                    self._cls_loss / self._count,
                    self._cls_distill_loss / self._count,
                    self._token_distill_loss / self._count,
                )
            )
            self._count = 0
            self._cls_loss = 0.0
            self._cls_distill_loss = 0.0
            self._token_distill_loss = 0.0

    def forward(self, inputs, outputs, labels):
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            raise ValueError(
                "StructuredPruningDistillLoss expects the model to return logits and token features."
            )

        pred = outputs[0]
        token_pred = outputs[1]

        cls_loss = self.base_criterion(pred, labels)

        need_teacher = (
            self.teacher_model is not None and (self.distill_weight > 0 or self.token_weight > 0)
        )
        if need_teacher:
            with torch.no_grad():
                cls_t, token_t = self.teacher_model(inputs)
        else:
            cls_t = None
            token_t = None

        if self.distill_weight > 0 and cls_t is not None:
            cls_kl_loss = F.kl_div(
                F.log_softmax(pred, dim=-1),
                F.log_softmax(cls_t, dim=-1),
                reduction="batchmean",
                log_target=True,
            )
        else:
            cls_kl_loss = pred.detach().new_tensor(0.0)

        if self.token_weight > 0 and token_t is not None:
            if self.mse_token:
                token_loss = torch.pow(token_pred - token_t, 2).mean()
            else:
                token_loss = F.kl_div(
                    F.log_softmax(token_pred, dim=-1),
                    F.log_softmax(token_t, dim=-1),
                    reduction="batchmean",
                    log_target=True,
                )
        else:
            token_loss = pred.detach().new_tensor(0.0)

        total_loss = (
            self.clf_weight * cls_loss
            + self.distill_weight * cls_kl_loss
            + self.token_weight * token_loss
        )

        self._log_progress(cls_loss.item(), cls_kl_loss.item(), token_loss.item())

        # Keep the metric layout expected by engine.train_one_epoch.
        zero = cls_loss.detach().new_tensor(0.0)
        loss_part = [
            cls_loss.detach(),
            zero.clone(),  # ratio placeholder
            cls_kl_loss.detach(),
            token_loss.detach(),
            zero.clone(),  # layer mse placeholder
            zero.clone(),  # feature kl placeholder
        ]
        return total_loss, loss_part
