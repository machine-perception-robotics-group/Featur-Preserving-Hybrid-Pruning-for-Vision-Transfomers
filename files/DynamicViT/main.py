# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#torchrun --nproc_per_node=1 --master_port=29500 main.py --model deit-b --input_size 224 --data_set CIFAR --data_path /home/kouki/datasets/cifar100 --nb_classes 100 --batch_size 128 --epochs 100 --lr 5e-4 --weight_decay 0.05 --drop_path 0.2 --base_rate 0.7 --ratio_weight 5.0 --finetune pretrained/deit_base_patch16_224-b5f2ef4d.pth --output_dir outputs/cifar100_deit-b --log_dir logs/cifar100_deit-b07
#torchrun --nproc_per_node=1 --master_port=29501 main.py --model deit-b --input_size 224 --data_set CIFAR --data_path /home/kouki/datasets/cifar100 --nb_classes 100 --batch_size 128 --epochs 150 --lr 5e-4 --weight_decay 0.05 --drop_path 0.2 --base_rate 0.5 --ratio_weight 5.0 --finetune pretrained/deit_base_patch16_224-b5f2ef4d.pth --output_dir outputs/token/base-rate50-re
import argparse
import datetime
import time
import torch
import torch.distributed as dist
import json
import os
import inspect

from pathlib import Path

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from optim_factory import create_optimizer

from datasets import build_dataset
from engine import train_one_epoch, evaluate

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from losses import DistillDiffPruningLoss_dynamic, TokenPruningClassificationLoss
from models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
from calc_flops import calc_flops, throughput
from structured_losses import StructuredPruningDistillLoss

import warnings
warnings.filterwarnings('ignore')


def load_checkpoint(path, map_location='cpu'):
    """Load checkpoints safely across PyTorch versions."""
    kwargs = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def infer_structured_mlp_dims(path):
    """
    Inspect a checkpoint to recover structured MLP hidden dimensions if present.
    Falls back to examining the block weights if explicit metadata is missing.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        checkpoint = load_checkpoint(path, map_location='cpu')
    except Exception as err:  # pragma: no cover - best effort helper
        print(f"Warning: unable to parse structured MLP dims from {path}: {err}")
        return None

    dims = checkpoint.get('structured_mlp_hidden_dims')
    if dims is not None:
        return [int(v) for v in dims]

    state_dict = checkpoint.get('model', checkpoint)
    if not isinstance(state_dict, dict):
        return None

    inferred = []
    idx = 0
    while True:
        key = f'blocks.{idx}.mlp.fc1.weight'
        tensor = state_dict.get(key)
        if tensor is None:
            break
        inferred.append(int(tensor.shape[0]))
        idx += 1
    return inferred or None

def get_args_parser():
    parser = argparse.ArgumentParser('Dynamic training script', add_help=False)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=utils.str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=utils.str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder', 'STANFORD_DOGS'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--gpu-id', default='',
                        help='Comma-separated GPU indices to expose via CUDA_VISIBLE_DEVICES when using CUDA.')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=utils.str2bool, default=False)
    parser.add_argument('--save_ckpt', type=utils.str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=utils.str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=utils.str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=utils.str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=utils.str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=utils.str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=utils.str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    parser.add_argument('--use_checkpoint', type=utils.str2bool, default=False,
                        help='Enable gradient checkpointing to trade compute for lower memory.')
    parser.add_argument('--teacher_path', default='',
                        help='Optional path to a teacher checkpoint (e.g., CIFAR fine-tuned).')
    parser.add_argument('--disable_distillation', action='store_true',
                        help='Skip teacher guidance and train token pruning with classification + ratio losses only.')

    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--lr_scale', type=float, default=0.01)
    parser.add_argument('--base_rate', type=float, default=0.9)
    parser.add_argument('--token-rate', type=float, nargs=3, default=None,
                        help='Explicit keep ratio (0-1) for each pruning stage')
    parser.add_argument('--ratio_weight', type=float, default=2.0)
    parser.add_argument('--report-flops', action='store_true',
                        help='Report GFLOPs for the configured model')
    parser.add_argument('--disable_reference_cifar_defaults', action='store_true',
                        help='Do not auto-apply the retraining settings collected from /home/kouki/DynamicViT when training on CIFAR-100.')
    parser.add_argument('--structured-mlp-only', action='store_true',
                        help='When true, train structured MLP pruning without token pruning losses.')

    return parser

DEFAULT_CIFAR_TEACHER = Path("outputs/baseline_cifar100-re/checkpoint-best.pth")



REFERENCE_CIFAR_SETTINGS = {
    "batch_size": 128,
    "epochs": 150,
    "lr": 1.25e-4,
    "warmup_epochs": 5,
    "drop_path": 0.1,
    "weight_decay": 0.05,
    "base_rate": 1.0,
    "ratio_weight": 2.0,
    "lr_scale": 0.01,
    "nb_classes": 100,
    "token_rate": [1.0, 1.0, 1.0],
}


def log_invocation(args):
    """
    Record the executed CLI command under output_dir/run_command.txt.
    """
    if not args.output_dir:
        return
    utils.log_cli_command(Path(args.output_dir) / "run_command.txt", only_main=True)


def apply_reference_retrain_settings(args, cli_overrides=None):
    """
    Align CIFAR-100 runs with the working retraining recipe stored in
    /home/kouki/DynamicViT unless explicitly disabled or the user supplied
    explicit CLI overrides.
    """
    if args.data_set.lower() != 'cifar' or args.disable_reference_cifar_defaults:
        return
    cli_overrides = set(cli_overrides or ())
    overrides = dict(REFERENCE_CIFAR_SETTINGS)
    if "token_rate" not in overrides:
        base_rate = overrides.get("base_rate")
        if base_rate is not None:
            overrides["token_rate"] = [base_rate, base_rate ** 2, base_rate ** 3]
    for key, value in overrides.items():
        if key in cli_overrides:
            continue
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if current != value:
            print(f"[reference retrain] {key}: {current} -> {value}")
            setattr(args, key, value)

def main(args, cli_overrides=None):
    apply_reference_retrain_settings(args, cli_overrides=cli_overrides)
    utils.init_distributed_mode(args)
    log_invocation(args)
    print(args)
    if args.device.startswith('cuda'):
        ids = [gpu.strip() for gpu in str(args.gpu_id).split(',') if gpu.strip()]
        if ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
            if args.device == 'cuda':
                args.device = 'cuda:1'
    device = torch.device(args.device)

    # fix the seed for reproducibility
    utils.set_global_seed(args.seed)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    base_criterion = criterion

    print(args.model)

    structured_mlp_dims = None
    structured_mlp_source = None
    if 'deit' in args.model:
        for candidate in (args.finetune, args.resume):
            dims = infer_structured_mlp_dims(candidate)
            if dims:
                structured_mlp_dims = dims
                structured_mlp_source = candidate
                print(f"Detected structured MLP hidden dims (len={len(dims)}) from {candidate}.")
                break

    custom_rate = list(args.token_rate) if args.token_rate is not None else None
    KEEP_RATE_TRANSFORMER = custom_rate if custom_rate is not None else [args.base_rate, args.base_rate ** 2, args.base_rate ** 3]

    teacher_model = None
    if args.model == 'deit-s':
        PRUNING_LOC = [3, 6, 9]
        KEEP_RATE = KEEP_RATE_TRANSFORMER
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, mlp_hidden_dims=structured_mlp_dims,
            use_checkpoint=args.use_checkpoint
        )
        pretrained = load_checkpoint('./pretrained/deit_small_patch16_224-cd65a155.pth', map_location='cpu')
        if not args.disable_distillation:
            teacher_model = VisionTransformerTeacher(
                patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
    elif args.model == 'deit-b':
        PRUNING_LOC = [3, 6, 9]
        KEEP_RATE = KEEP_RATE_TRANSFORMER
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, drop_path_rate=args.drop_path,
            mlp_hidden_dims=structured_mlp_dims, use_checkpoint=args.use_checkpoint
        )
        pretrained = load_checkpoint('./pretrained/deit_base_patch16_224-b5f2ef4d.pth', map_location='cpu')
        if not args.disable_distillation:
            teacher_model = VisionTransformerTeacher(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
    if 'convnext' in args.model or 'deit' in args.model or 'swin' in args.model:
        pretrained = pretrained['model']

    teacher_payload = None
    teacher_payload_source = None
    dataset_name = args.data_set.lower()
    if args.disable_distillation and args.teacher_path:
        print("[distill] --teacher_path is ignored because distillation is disabled.")
    if not args.disable_distillation:
        if args.teacher_path:
            teacher_candidate = Path(args.teacher_path)
            if teacher_candidate.is_file():
                teacher_ckpt = load_checkpoint(str(teacher_candidate), map_location='cpu')
                teacher_payload = teacher_ckpt.get('model', teacher_ckpt)
                teacher_payload_source = str(teacher_candidate)
            else:
                raise FileNotFoundError(f"[teacher] Specified teacher checkpoint {teacher_candidate} does not exist.")
        elif dataset_name == 'cifar':
            if not DEFAULT_CIFAR_TEACHER.is_file():
                raise FileNotFoundError(
                    f"[teacher] CIFAR runs require {DEFAULT_CIFAR_TEACHER}, but the file is missing."
                )
            teacher_ckpt = load_checkpoint(str(DEFAULT_CIFAR_TEACHER), map_location='cpu')
            teacher_payload = teacher_ckpt.get('model', teacher_ckpt)
            teacher_payload_source = str(DEFAULT_CIFAR_TEACHER)
        if teacher_payload is not None:
            print(f"[teacher] Loaded teacher checkpoint from {teacher_payload_source}")
    skip_student_pretrained = (
        'deit' in args.model and structured_mlp_source is not None and structured_mlp_dims is not None
    )
    if skip_student_pretrained:
        print(f"Skip loading default pretrained weights into student; will load structured checkpoint from {structured_mlp_source}.")
    else:
        utils.load_state_dict(model, pretrained)

    if args.nb_classes != 1000 and hasattr(model, 'reset_classifier'):
        model.reset_classifier(args.nb_classes)

    if not args.disable_distillation:
        teacher_state_dict = teacher_payload if teacher_payload is not None else pretrained
        # align classifier head size before loading teacher weights if checkpoint already matches nb_classes
        teacher_head = teacher_state_dict.get('head.weight') if isinstance(teacher_state_dict, dict) else None
        if teacher_head is not None and teacher_head.shape[0] == args.nb_classes:
            teacher_model.reset_classifier(args.nb_classes)
        utils.load_state_dict(teacher_model, teacher_state_dict)
        if args.nb_classes != 1000 and hasattr(teacher_model, 'reset_classifier'):
            reset_teacher_head = True
            if isinstance(teacher_state_dict, dict):
                head_weight = teacher_state_dict.get('head.weight')
                if head_weight is not None and head_weight.shape[0] == args.nb_classes:
                    reset_teacher_head = False
            if reset_teacher_head:
                teacher_model.reset_classifier(args.nb_classes)
        teacher_model.eval()
        teacher_model = teacher_model.to(device)
        print('success load teacher model weight')
    if args.disable_distillation:
        if 'deit' in args.model:
            if args.structured_mlp_only:
                raise ValueError("--structured-mlp-only requires distillation; disable one of the options.")
            criterion = TokenPruningClassificationLoss(
                base_criterion=base_criterion,
                clf_weight=1.0,
                pruning_loc=PRUNING_LOC,
                keep_ratio=KEEP_RATE,
                ratio_weight=args.ratio_weight,
            )
        else:
            raise ValueError("Distillation-free training is currently supported only for DeiT-based token pruning.")
    else:
        if 'deit' in args.model:
            if args.structured_mlp_only:
                criterion = StructuredPruningDistillLoss(
                    teacher_model=teacher_model,
                    base_criterion=base_criterion,
                    clf_weight=1.0,
                    distill_weight=0.5,
                    token_weight=0.0,
                    mse_token=True,
                    print_mode=True,
                )
            else:
                criterion = DistillDiffPruningLoss_dynamic(
                    teacher_model, base_criterion, clf_weight=1.0, keep_ratio=KEEP_RATE, mse_token=True, ratio_weight=args.ratio_weight, distill_weight=0.5
                )

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
    print('number of params:', params_in_millions)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = load_checkpoint(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    if args.throughput:
        if utils.is_main_process():
            print('# throughput test')
            image = torch.randn(args.batch_size, 3, args.input_size, args.input_size)
            throughput(image, model)
            del image
        if utils.is_dist_avail_and_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    model_without_ddp = model
    n_parameters = total_params

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    samples_per_epoch = len(dataset_train)
    print("Number of training examples = %d" % samples_per_epoch)
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=None,
        get_layer_scale=None,
        bone_lr_scale=args.lr_scale)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))


    print("criterion = %s" % str(criterion))

    max_accuracy = 0.0
    model_ema = None
    max_accuracy, max_accuracy_ema = utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )
        if utils.is_main_process() and args.report_flops and flops_value is not None:
            print(f"[Epoch {epoch}] FLOPs (GFLOPs): {flops_value:.4f}")

        args.epoch_throughput = True
        if utils.is_main_process() and args.epoch_throughput:
            print(f"[Epoch {epoch}] throughput test")
            image = torch.randn(args.batch_size, 3, args.input_size, args.input_size)
            prev_state = model_without_ddp.training
            throughput(image, model_without_ddp)
            if prev_state:
                model_without_ddp.train(True)
            del image

        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema, best_acc=max_accuracy, best_acc_ema=max_accuracy_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

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

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, best_acc=max_accuracy, best_acc_ema=max_accuracy_ema)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic training script', parents=[get_args_parser()])
    default_args = parser.parse_args([])
    args = parser.parse_args()
    cli_overrides = {
        key for key, value in vars(args).items()
        if getattr(default_args, key, None) != value
    }
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, cli_overrides=cli_overrides)
