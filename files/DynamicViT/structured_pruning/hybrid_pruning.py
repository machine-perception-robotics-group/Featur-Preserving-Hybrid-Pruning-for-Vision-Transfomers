#実行文　python3 structured_pruning/experiment_cls_grad_prune.py --model deit-b --input_size 224 --data_set CIFAR --data_path /home/kouki/datasets/cifar100 --nb_classes 100 --batch_size 128  --lr 5e-4 --weight_decay 0.05 --drop_path 0.2 --base_rate 0.7 --ratio_weight 5.0 --finetune pretrained/deit_base_patch16_224-b5f2ef4d.pth --cls-sim-ref outputs/token/base-rate70/checkpoint-best.pth --cls-sim-weight 0.05 --token-train-epochs 120 --grad-collect-epochs 1 --struct-keep-ratio 0.5 --struct-min-channels 96 --finetune-epochs 80 --experiment-root outputs/cls_grad_pipeline

#!/usr/bin/env python3
"""
Pipeline script for:
1) DynamicViT token pruning with CLS cosine similarity regularization.
2) Gradient-based MLP structured pruning using accumulated training gradients.
3) Fine-tuning the structurally-pruned model.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

sys.path.append('.')
from datasets import build_dataset
from engine import evaluate
from losses import DistillDiffPruningLoss_dynamic
from main import (
    apply_reference_retrain_settings,
    get_args_parser,
    infer_structured_mlp_dims,
    load_checkpoint,
    log_invocation,
)
from optim_factory import create_optimizer
import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from calc_flops import calc_flops, throughput
from compare_cls_token_similarity import calc_cls_similarity

from structured_pruning.train_with_gate_pruning import (
    ClsSimilarityRegularizer,
    GradientPruner,
    build_student_model,
    cosine_scheduler,
    load_teacher,
    train_one_epoch_with_gates,
)


@dataclass
class StageResult:
    model: torch.nn.Module
    best_acc: float
    checkpoint_path: Path
    grad_pruner: Optional[GradientPruner] = None


def parse_pipeline_args():
    parser = argparse.ArgumentParser(
        "DynamicViT cosine + gradient structured pruning experiment",
        parents=[get_args_parser()],
    )
    parser.add_argument("--experiment-root", required=True,
                        help="Root directory where stage outputs/checkpoints are written.")
    parser.add_argument("--token-train-epochs", type=int, default=120,
                        help="Epochs for the token-pruned training stage.")
    parser.add_argument("--finetune-epochs", type=int, default=80,
                        help="Epochs for the post-pruning fine-tuning stage.")
    parser.add_argument("--finetune-layers", type=int, nargs='+', default=None,
                        help="Layer indices where finetuning is performed.")
    parser.add_argument("--update-only-ffn", action='store_true',
                        help="Update ffn in ViT layers.")
    parser.add_argument("--update-classifier", action='store_true',
                        help="Update classifier layer even when update layers specified.")
    parser.add_argument("--grad-collect-epochs", type=int, default=5,
                        help="Number of ending epochs that accumulate gradients for structured pruning.")
    parser.add_argument("--struct-keep-ratio", type=float, default=0.6,
                        help="Keep ratio when converting gradients into structured pruning decisions.")
    parser.add_argument("--struct-prune-layers", type=int, nargs='+', default=None,
                        help="Layer indices where structured pruning is performed.")
    parser.add_argument("--struct-min-channels", type=int, default=96,
                        help="Minimum hidden width to retain inside each block after pruning.")
    parser.add_argument("--cls-sim-ref", default="",
                        help="Reference checkpoint for CLS cosine regularization.")
    parser.add_argument("--cls-sim-weight", type=float, default=0.0,
                        help="Weight for the CLS cosine similarity penalty.")
    parser.add_argument("--cls-sim-layers", type=int, nargs='+', default=None,
                        help="Layer indices used for the CLS similarity loss.")
    parser.add_argument("--cls-sim-feature", choices=["cls", "tokens"], default="cls",
                        help="Feature slice evaluated by the CLS regularizer.")
    parser.add_argument("--cls-sim-loss", choices=["cosine", "l2", "cosine_l2"], default="cosine",
                        help="Loss type applied by the CLS regularizer.")
    parser.add_argument("--calc-cls-sim", action='store_true',
                        help="Calculate the cls similarity during training.")
    parser.add_argument("--prune-grad-L-cls", action='store_true',
                        help="Use gradient by L_cls for MLP pruning.")
    parser.add_argument("--distill-weight", type=float, default=0.5,
                        help="Weight for the distill penalty.")
    return parser


def clone_args(base: argparse.Namespace) -> argparse.Namespace:
    return copy.deepcopy(base)


def prepare_dataloaders(args: argparse.Namespace):
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_eval, _ = build_dataset(is_train=False, args=args)
    if args.distributed:
        sampler_train = DistributedSampler(
            dataset_train,
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank(),
            shuffle=True,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_eval)
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_eval,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return dataset_train, data_loader_train, data_loader_val


def maybe_build_mixup(args: argparse.Namespace) -> Optional[Mixup]:
    if args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None:
        return Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )
    return None


def load_student_weights(model: torch.nn.Module, args: argparse.Namespace):
    if not args.finetune:
        return
    checkpoint = load_checkpoint(args.finetune, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    current = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in state and key in current and state[key].shape != current[key].shape:
            state.pop(key)
    utils.load_state_dict(model, state)


def build_student_and_teacher(args: argparse.Namespace, device: torch.device):
    ref_dims = infer_structured_mlp_dims(args.finetune) if args.finetune else None
    model = build_student_model(args, mlp_hidden_dims=ref_dims)
    load_student_weights(model, args)
    teacher_model = load_teacher(args, device)
    return model, teacher_model


def build_base_criterion(args: argparse.Namespace):
    if args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None:
        return SoftTargetCrossEntropy()
    if args.smoothing > 0.0:
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    return torch.nn.CrossEntropyLoss()


def build_distill_criterion(args: argparse.Namespace, teacher_model, use_only_loss_cls=False):
    if args.token_rate:
        keep_ratio = args.token_rate
    else:
        keep_ratio = [args.base_rate, args.base_rate ** 2, args.base_rate ** 3],

    if use_only_loss_cls:
        clf_weight = 0.0
        ratio_weight = 0.0
        distill_weight = 0.0
    else:
        clf_weight = 1.0
        ratio_weight = args.ratio_weight
        distill_weight = args.distill_weight

    return DistillDiffPruningLoss_dynamic(
        teacher_model=teacher_model,
        base_criterion=build_base_criterion(args),
        clf_weight=clf_weight,
        keep_ratio=keep_ratio,
        mse_token=True,
        ratio_weight=ratio_weight,
        distill_weight=distill_weight,
    )


def capture_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    module = model.module if hasattr(model, "module") else model
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def save_checkpoint(path: Path, model: torch.nn.Module, extra: Optional[Dict] = None):
    module = model.module if hasattr(model, "module") else model
    payload = {"model": {k: v.cpu() for k, v in module.state_dict().items()}}
    dims = getattr(module, "mlp_hidden_dims", None)
    if dims is not None:
        payload["structured_mlp_hidden_dims"] = [int(v) for v in dims]
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def maybe_build_cls_regularizer(args: argparse.Namespace, model: torch.nn.Module, device: torch.device):
    if args.cls_sim_weight <= 0 or not args.cls_sim_ref:
        return None
    # ref_dims = infer_structured_mlp_dims(args.cls_sim_ref)
    # reference_model = build_student_model(args, mlp_hidden_dims=ref_dims)
    # REVIEW: COS類似度で参照するモデルは、teacherモデルでよいのでは？
    # studentモデルだと、トークンプルーニングされたものが使われてしまう
    reference_model = load_teacher(args, device)
    checkpoint = load_checkpoint(args.cls_sim_ref, map_location="cpu")
    ref_state = checkpoint.get("model", checkpoint)
    utils.load_state_dict(reference_model, ref_state)
    return ClsSimilarityRegularizer(
        student=model,
        reference=reference_model,
        layers=args.cls_sim_layers,
        weight=args.cls_sim_weight,
        device=device,
        feature_type=args.cls_sim_feature,
        loss_type=args.cls_sim_loss,
    )


def run_training_stage(
    args: argparse.Namespace,
    stage_name: str,
    epochs: int,
    output_dir: Path,
    collect_gradients: bool,
    grad_collect_epochs: int,
    finetune_layers: list[int] = None,
    update_only_ffn: bool = False,
    update_classifier: bool = False,
    only_grad_accumulate: bool = False,
    use_only_loss_cls: bool = False,
) -> StageResult:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir = str(output_dir)
    args.log_dir = str(output_dir / "logs")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    args.epochs = epochs
    stage_cmd = " ".join(shlex.quote(token) for token in sys.argv)
    utils.log_cli_command(
        Path(args.output_dir) / "run_command.txt",
        command=f"{stage_name}: {stage_cmd}",
        only_main=True,
    )

    grad_collect_epochs = max(0, min(grad_collect_epochs, epochs))
    dataset_train, data_loader_train, data_loader_val = prepare_dataloaders(args)
    mixup_fn = maybe_build_mixup(args)

    model, teacher_model = build_student_and_teacher(args, device)

    model.eval()
    flops_value = None
    if utils.is_main_process():
        try:
            flops_value = calc_flops(model, args.input_size)
            print('FLOPs:', flops_value)
        except Exception as err:
            print(f"[FLOPs] Failed to compute: {err}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_millions = total_params / 1e6
    samples_per_epoch = len(dataset_train)

    model.to(device)
    teacher_model.to(device)

    criterion = build_distill_criterion(args, teacher_model, use_only_loss_cls)
    model_ema = None
    optimizer = create_optimizer(args, model, skip_list=None)
    loss_scaler = NativeScaler()

    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    lr_schedule = cosine_scheduler(
        args.lr,
        args.min_lr,
        epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end or args.weight_decay,
        epochs,
        num_training_steps_per_epoch,
    )

    cls_regularizer = maybe_build_cls_regularizer(args, model, device)
    grad_pruner = GradientPruner(model, min_channels=args.struct_min_channels) if collect_gradients else None

    best_acc = 0.0
    best_state = None
    log_path = Path(args.output_dir) / f"{stage_name}_log.jsonl"
    max_accuracy = 0.0
    start_time = time.time()

    # 全てのパラメータを更新するように初期化
    for n, p in model.named_parameters():
        p.requires_grad = True

    # パラメータ更新を行う層の設定
    # REVIEW: かなり複雑になっている
    if finetune_layers is not None:
        # 対象外の層の重みをフリーズ。
        # トークンプルーニングが行われるのは、l={3, 6, 9}
        score_pred_id_to_l = {3: 0, 6: 1, 9: 2}
        for n, p in model.named_parameters():
            is_freeze = True

            if update_only_ffn & ('mlp.fc' not in n):
                # ViTのFFNのみパラメータ更新するようにする
                p.requires_grad = False
                continue

            for l_id in finetune_layers:
                # Transformerブロック
                if f"blocks.{l_id}." in n:
                    is_freeze = False
                    print('is_freeze = False', n)
                    break
                # トークン枝刈りを行うscore_predictor
                if (l_id in score_pred_id_to_l.keys()) and (f"score_predictor.{score_pred_id_to_l[l_id]}." in n):
                    is_freeze = False
                    print('is_freeze = False', n)
                    break
            if is_freeze:
                p.requires_grad = False
    else:
        for n, p in model.named_parameters():
            if update_only_ffn & ('mlp.fc' not in n):
                # ViTのFFNのみパラメータ更新するようにする
                p.requires_grad = False

    # classifierをパラメータ更新するようにする
    if update_classifier:
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True

    for n, p in model.named_parameters():
        print(f'requires_grad={p.requires_grad}', n)

    if args.calc_cls_sim:
        reference_model = load_teacher(args, device)
        checkpoint = load_checkpoint(args.cls_sim_ref, map_location="cpu")
        ref_state = checkpoint.get("model", checkpoint)
        utils.load_state_dict(reference_model, ref_state)

        save_similarity_path = Path(args.output_dir) / "cls_similarity_init.json"
        calc_cls_similarity(reference_model, model, data_loader_val, device, save_similarity_path)

    for epoch in range(args.start_epoch, args.start_epoch + epochs):
        epoch_start = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        should_collect = (
            grad_pruner is not None and
            epoch >= (args.start_epoch + epochs - grad_collect_epochs)
        )
        grad_tracker = grad_pruner if should_collect else None

        train_stats = train_one_epoch_with_gates(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            cls_regularizer=cls_regularizer,
            grad_tracker=grad_tracker,
            only_grad_accumulate=only_grad_accumulate,
        )

        args.epoch_throughput = True
        if utils.is_main_process() and args.epoch_throughput:
            print(f"[Epoch {epoch}] throughput test")
            image = torch.randn(args.batch_size, 3, args.input_size, args.input_size)
            prev_state = model.training
            throughput(image, model)
            if prev_state:
                model.train(True)
            del image

        test_stats = evaluate(data_loader_val, model, device)

        if args.calc_cls_sim:
            save_similarity_path = Path(args.output_dir) / f"cls_similarity_epoch_{epoch}.json"
            calc_cls_similarity(reference_model, model, data_loader_val, device, save_similarity_path)

        acc1 = float(test_stats["acc1"])
        max_accuracy = max(max_accuracy, acc1)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "best": max_accuracy,
            "stage": stage_name,
        }

        epoch_time = time.time() - epoch_start
        throughput_per_epoch = samples_per_epoch / epoch_time if epoch_time > 0 else 0.0
        if utils.is_main_process():
            stats_msg = f"[Epoch {epoch}] Params: {params_in_millions:.3f}M, Throughput: {throughput_per_epoch:.2f} img/s"
            if flops_value is not None:
                stats_msg += f", FLOPs: {flops_value:.4f} GFLOPs"
            print(stats_msg)

        log_stats['params_m'] = params_in_millions
        log_stats['throughput'] = throughput_per_epoch
        if flops_value is not None:
            log_stats['flops'] = flops_value

        if utils.is_main_process():
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_stats) + "\n")

        if acc1 >= best_acc - 1e-5:
            best_acc = acc1
            best_state = capture_model_state(model)

    total_time = time.time() - start_time
    print(f"[{stage_name}] Finished in {total_time / 60:.1f} minutes. Best Acc@1 {best_acc:.2f}%")

    module = model.module if hasattr(model, "module") else model
    if best_state is not None:
        module.load_state_dict(best_state)
    checkpoint_path = Path(args.output_dir) / "checkpoint-best.pth"
    save_checkpoint(checkpoint_path, module, extra={"best_acc": best_acc})
    return StageResult(
        model=module,
        best_acc=best_acc,
        checkpoint_path=checkpoint_path,
        grad_pruner=grad_pruner,
    )


# 勾配ベースの構造化枝刈り
def apply_gradient_pruning(
    model: torch.nn.Module,
    grad_pruner: GradientPruner,
    keep_ratio: float,
    prune_layers: list[int],
    min_channels: int,
    prune_dir: Path,
) -> Path:
    prune_dir.mkdir(parents=True, exist_ok=True)
    grad_pruner.min_channels = min_channels
    grad_pruner.prune(keep_ratio, prune_layers)
    model.mlp_hidden_dims = [block.mlp.fc1.out_features for block in model.blocks]
    checkpoint_path = prune_dir / "checkpoint-grad-pruned.pth"
    save_checkpoint(
        checkpoint_path,
        model,
        extra={
            "grad_keep_ratio": keep_ratio,
            "structured_mlp_hidden_dims": model.mlp_hidden_dims,
        },
    )
    structure_log = prune_dir / "structure.txt"
    with structure_log.open("w", encoding="utf-8") as handle:
        for line in grad_pruner.describe_structure():
            handle.write(line + "\n")
    print(f"[grad-prune] Saved structured checkpoint to {checkpoint_path}")
    return checkpoint_path


def run_pipeline(args: argparse.Namespace):
    utils.init_distributed_mode(args)
    log_invocation(args)
    utils.set_global_seed(args.seed)

    root = Path(args.experiment_root)
    utils.log_cli_command(root / "run_command.txt", only_main=True)
    token_dir = root / "token_train"
    grad_dir = root / "collect_grad"
    prune_dir = root / "structured_prune"
    finetune_dir = root / "finetune"

    # トークン枝刈り
    token_args = clone_args(args)
    token_args.start_epoch = 0
    if args.token_train_epochs > 0:
        token_stage = run_training_stage(
            token_args,
            stage_name="token_train",
            epochs=args.token_train_epochs,
            output_dir=token_dir,
            collect_gradients=False,
            grad_collect_epochs=0,
            finetune_layers=args.finetune_layers,
            update_only_ffn=args.update_only_ffn,
            update_classifier=args.update_classifier
        )
    else:
        token_stage = StageResult(
            model=None,
            best_acc=-1,
            checkpoint_path='',
            grad_pruner=None,
        )

    # 勾配の収集
    collect_grad_args = clone_args(args)
    collect_grad_args.start_epoch = 0
    if token_stage.checkpoint_path:
        collect_grad_args.finetune = str(token_stage.checkpoint_path)
    print(f'use "{collect_grad_args.finetune}" to collect gradients.')
    use_only_loss_cls = False
    if args.prune_grad_L_cls:
        use_only_loss_cls = True
    collect_grad_stage = run_training_stage(
        collect_grad_args,
        stage_name="collect_grad",
        epochs=args.grad_collect_epochs,
        output_dir=grad_dir,
        collect_gradients=True,
        use_only_loss_cls=use_only_loss_cls,
        only_grad_accumulate=True,
        grad_collect_epochs=args.grad_collect_epochs,
    )

    # 構造化枝刈り
    if collect_grad_stage.grad_pruner is None:
        raise RuntimeError("Gradient tracker was not initialized during token training.")
    structured_ckpt = apply_gradient_pruning(
        collect_grad_stage.model,
        collect_grad_stage.grad_pruner,
        keep_ratio=args.struct_keep_ratio,
        prune_layers=args.struct_prune_layers,
        min_channels=args.struct_min_channels,
        prune_dir=prune_dir,
    )

    # ファインチューニング
    finetune_args = clone_args(args)
    finetune_args.finetune = str(structured_ckpt)
    finetune_args.start_epoch = 0
    if args.finetune_epochs > 0:
        finetune_stage = run_training_stage(
            finetune_args,
            stage_name="finetune",
            epochs=args.finetune_epochs,
            output_dir=finetune_dir,
            collect_gradients=False,
            grad_collect_epochs=0,
            finetune_layers=args.finetune_layers,
            update_only_ffn=args.update_only_ffn,
            update_classifier=args.update_classifier
        )
    else:
        finetune_stage = StageResult(
            model=None,
            best_acc=-1,
            checkpoint_path='',
            grad_pruner=None,
        )

    summary = {
        "token_stage": {
            "best_acc": token_stage.best_acc,
            "checkpoint": str(token_stage.checkpoint_path),
        },
        "structured_prune": {
            "checkpoint": str(structured_ckpt),
            "keep_ratio": args.struct_keep_ratio,
        },
        "finetune_stage": {
            "best_acc": finetune_stage.best_acc,
            "checkpoint": str(finetune_stage.checkpoint_path),
        },
    }
    summary_path = root / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[summary] Wrote experiment summary to {summary_path}")


def main():
    parser = parse_pipeline_args()
    args = parser.parse_args()

    cli_overrides = {
        key for key in vars(args).keys()
        if parser.get_default(key) != getattr(args, key)
    }
    remembered_token_rate = args.token_rate
    remembered_base_rate = args.base_rate
    apply_reference_retrain_settings(args, cli_overrides=cli_overrides)
    if "token_rate" in cli_overrides:
        args.token_rate = remembered_token_rate
    else:
        args.token_rate = None
    if "base_rate" in cli_overrides:
        args.base_rate = remembered_base_rate

    if not args.finetune:
        raise ValueError("Specify --finetune with the pretrained token-pruned checkpoint.")

    run_pipeline(args)


if __name__ == "__main__":
    main()
