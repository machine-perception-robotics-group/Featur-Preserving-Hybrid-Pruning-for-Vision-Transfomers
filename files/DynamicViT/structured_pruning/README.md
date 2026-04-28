# Structured MLP Pruning (Norm-Based)

This folder provides utilities to run structured pruning on the MLP layers of `VisionTransformerDiffPruning`
using either channel norms or a distribution-matching importance score. The goal is to physically remove hidden
channels with the lowest importance, reconstruct a smaller model, and continue fine-tuning on CIFAR-100.

> **Note:** The original norm-only implementation is preserved in `structured_pruning/mlp_pruning_norm.py`
> for reference or standalone use.

## Steps

1. **Prune the pretrained checkpoint**

   ```bash
   python structured_pruning/run_mlp_pruning.py \
     --checkpoint-in pretrained/deit_base_patch16_224-b5f2ef4d.pth \
     --checkpoint-out outputs/pruned_mlp/checkpoint-pruned.pth \
     --prune-ratio 0.3 \
     --num-classes 100 \
     --importance l1 \
     --device cuda
   ```

   - `--keep-ratio` controls the fraction of hidden channels kept per MLP block.
   - `--min-channels` (default 96) ensures each block retains a reasonable width.
   - `--importance l1` switches to the magnitude criterion used by Torch-Pruning. Keep the default (`l2`) to replicate the original DynamicViT heuristic.
   - Set `--importance kl` to estimate channel scores that keep the structured model close to the token-pruned teacher output distribution (see below).
   - The exported checkpoint stores `structured_mlp_hidden_dims`, enabling `main.py` to rebuild the exact architecture.

### Matching the token-pruned output distribution

When `--importance kl` is selected, the script inserts differentiable gates after each fc1 layer and uses a calibration
set to minimize the KL divergence between the current model and a teacher (by default the same checkpoint before pruning).
Required arguments for this mode:

- `--data-set` and `--data-path` (or `--eval-data-path` for arbitrary folders) describe the calibration data.
- `--calib-batch-size` and `--calib-batches` control how many samples are used to approximate the logits distribution.
- Optionally pass `--teacher-checkpoint` if the teacher differs from `--checkpoint-in`.

Example:

```bash
python structured_pruning/run_mlp_pruning.py \
  --checkpoint-in outputs/token_pruned/checkpoint.pth \
  --checkpoint-out outputs/pruned_mlp/kl_pruned.pth \
  --prune-ratio 0.3 \
  --importance kl \
  --data-set CIFAR \
  --data-path /home/kouki/datasets/cifar100 \
  --calib-batch-size 128 \
  --calib-batches 100
```

この `kl` 重要度ではトークン枝刈り済みモデル（教師）の出力分布に近づくようにチャンネルを削減するため、
単純なノルム基準よりも精度を維持しやすくなります。

2. **Fine-tune on CIFAR-100**

   Use `main.py` with the pruned checkpoint as the initialization:

   ```bash
   python main.py \
     --model deit-b \
     --data_set CIFAR \
     --data_path /home/kouki/datasets/cifar100 \
     --nb_classes 100 \
     --batch_size 128 \
     --epochs 100 \
     --finetune outputs/pruned_mlp/checkpoint-pruned.pth \
     --output_dir outputs/cifar100_deit-b_pruned \
     --log_dir logs/cifar100_deit-b_pruned
   ```

   All other hyper-parameters (lr, weight decay, token pruning ratios, etc.) can follow your existing setup.

`--prune-ratio` で削減率を直接指定します（例: 0.3 なら各 MLP で 30% 削除、70% を保持）。実行時に `--dry-run` を付けなければ、減らした構造のまま `structured_mlp_hidden_dims` 付きのチェックポイントが保存されます。

## 比較: 枝刈り前後の logit / 確率分布

トークン枝刈りモデルと構造化枝刈りモデルなど、任意の 2 つのチェックポイントの最終出力分布を比較したい場合は
`tools/compare_logits_distribution.py` を利用できます。

```bash
python tools/compare_logits_distribution.py \
  --before-checkpoint outputs/token/base-rate70/checkpoint-best.pth \
  --after-checkpoint outputs/structured_gate/checkpoint-best.pth \
  --model deit-b \
  --data_set CIFAR \
  --data_path /home/kouki/datasets/cifar100 \
  --nb_classes 100 \
  --batch_size 128 \
  --base_rate 0.7 \
  --save-summary logs/dist_summary.json \
  --save-details logs/dist_details.pt
```

スクリプトは以下をレポートします:
- `KL(before || after)`, `KL(after || before)`、Jensen-Shannon divergence
- 確率ベクトル／logit 差分の L1/L2/MSE とコサイン類似度
- Top-1 / Top-5 の精度と top-1 予測一致率
- 任意で per-sample の詳細指標 (`--save-details`)

`--max-samples` や `--max-batches` で評価サンプルを絞ることもできます。JSON summary を残したい場合は `--save-summary` を利用してください。
