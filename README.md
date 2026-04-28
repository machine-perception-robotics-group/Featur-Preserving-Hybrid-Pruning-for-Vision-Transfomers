#特徴表現を維持するViTのハイブリッド枝刈り：トークン枝刈りと構造化枝刈りの逐次交互最適化

本リポジトリでは、Vision Transformer に対してDyanamicViTをベースとしたトークン枝刈りと構造化枝刈りのハイブリッド化を行う
これによりトークン削減と、MLP層に対するパラメータ削減によってモデルの軽量化と精度維持の両立を行う。

---

## 環境

- Python
- PyTorch
- torchvision
- timm
- tensorboardX
- six
- fvcore

---

## データセット

- CIFAR-100

---

## 1. セットアップ

### 学習済みモデルのダウンロード

以下の学習済みモデルをダウンロードし、`DynamicViT/pretrained/` フォルダに格納します。

```text
https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
```

### Dockerイメージの作成

`docker/build/` フォルダで以下を実行します。

```bash
./build_image.sh
```

### Dockerコンテナの起動

`docker/` フォルダで以下を実行します。

```bash
./run_container.sh
```

新しい端末でコンテナに接続する場合は、`docker/` フォルダで以下を実行します。

```bash
./exec.sh
```

---

## 2. ベースモデルの作成

コンテナ内で `/workspace/files/DynamicViT/` フォルダに移動し、以下のコマンドを実行します。

```bash
python main.py \
  --model deit-b \
  --input_size 224 \
  --data_set CIFAR \
  --nb_classes 100 \
  --batch_size 64 \
  --epochs 150 \
  --lr 2.5e-4 \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --token-rate 1.0 1.0 1.0 \
  --base_rate 1.0 \
  --ratio_weight 5.0 \
  --finetune pretrained/deit_base_patch16_224-b5f2ef4d.pth \
  --disable_distillation \
  --output_dir outputs/baseline_cifar100-re
```

結果は以下のフォルダに保存されます。

```text
outputs/baseline_cifar100-re/
```

---

## 3. 交互段階的ハイブリッド枝刈り（L_clsあり）

以下のコマンドで、トークン枝刈りとMLP構造枝刈りを組み合わせたハイブリッド枝刈りを実行します。

```bash
python3 structured_pruning/hybrid_pruning.py \
  --model deit-b \
  --input_size 224 \
  --data_set CIFAR \
  --data_path ./data \
  --nb_classes 100 \
  --batch_size 64 \
  --lr 2.5e-4 \
  --warmup_steps 500 \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --token-rate 0.7 0.7 0.7 \
  --ratio_weight 5.0 \
  --finetune outputs/baseline_cifar100-re/checkpoint-best.pth \
  --cls-sim-ref outputs/baseline_cifar100-re/checkpoint-best.pth \
  --cls-sim-weight 0.05 \
  --cls-sim-layers 3 \
  --token-train-epochs 20 \
  --grad-collect-epochs 1 \
  --struct-keep-ratio 0.5 \
  --struct-prune-layers 0 1 2 3 \
  --struct-min-channels 96 \
  --finetune-epochs 20 \
  --finetune-layers 0 1 2 3 \
  --teacher_path outputs/baseline_cifar100-re/checkpoint-best.pth \
  --prune-grad-L-cls \
  --experiment-root outputs/hybrid_pruning_b_with_loss_cls_type2_step1
```

他のコマンド例は以下に記載しています。

```text
/workspace/files/DynamicViT/run_pruning.sh
```

---

## 4. 類似度計算

枝刈り前後のモデルを比較する場合は、`compare_cls_token_similarity.py` を使用します。

```bash
python3 compare_cls_token_similarity.py \
  --data-set CIFAR \
  --data-path ./data \
  --nb-classes 100 \
  --model deit-b \
  --batch-size 16 \
  --num-workers 0 \
  --before-checkpoint <path/to/reference_model> \
  --after-checkpoint <path/to/target_model> \
  --token-rate <例: 0.7 0.49 0.343> \
  --capture-all-blocks \
  --capture-points norm \
  --extra-token-metrics \
  --per-class-topk 0
```

`--token-rate` には、比較対象となるターゲットモデルのトークン枝刈り率を指定します。


## 実験内容

本リポジトリでは、主に以下の実験を行います。

- CIFAR-100におけるベースモデルの作成
- DynamicViTによるトークン枝刈り
- MLP層の構造的枝刈り
- 交互段階的ハイブリッド枝刈り
- 枝刈り前後のCLSトークン類似度評価
- 枝刈り前後のlogit / 確率分布比較

---

## 参考にしたリポジトリ

- DynamicViT  
  https://github.com/raoyongming/DynamicViT
