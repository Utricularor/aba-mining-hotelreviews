# 対照学習を加えた性能比較実験方針

## 実験目的

最適化済みハイパーパラメータを使用した3つのモデルにおいて、対照学習による埋め込み最適化が性能に与える影響を評価する。

## 実験設計

### 比較対象

**モデル（3つ）:**
1. **FreezedBertRgcnMlp** - グラフベースモデル（R-GCN + 固定BERT埋め込み）
2. **FreezedBertMlp** - 固定BERT + MLP分類層
3. **FinetunedBertMlp** - 微調整BERT + MLP分類層

**条件（2つ）:**
- **対照学習なし（ベースライン）**: 通常のBERT埋め込みを使用
- **対照学習あり（提案手法）**: 対照学習で最適化した埋め込みを使用

**合計: 3モデル × 2条件 = 6実験設定**

### 実験ID

- 対照学習なし: `exp_25_1106_baseline_named_models`
- 対照学習あり: `exp_25_1106_contrastive_named_models`

## 実装方針

### 1. `run_named_models.py`の拡張

- `--use-contrastive`オプションを追加
- 対照学習ありの場合、`apply_contrastive_learning`関数を呼び出し
- 最適化された埋め込みで`embedding_matrix`と`data.x`を更新

### 2. 最適化済みハイパーパラメータの使用

各モデルのスイープ結果から最良設定を抽出：

#### FreezedBertRgcnMlp
- スイープ結果からF1平均が最大の設定を使用
- 例: `FreezedBertRgcnMlp[hd=64,layers=2,dr=0.1,lr=0.0005,ep=60]`

#### FreezedBertMlp
- スイープ結果からF1平均が最大の設定を使用
- 例: `FreezedBertMlp[lr=0.001,do=0.2,ml=128,bs=16]`

#### FinetunedBertMlp
- スイープ結果からF1平均が最大の設定を使用
- 例: `FinetunedBertMlp[lr=3e-05,do=0.2,ml=128,bs=32]`

### 3. 対照学習設定

`config/robust_experiment.yaml`の設定を使用：
```yaml
contrastive_learning:
  enabled: true
  hidden_dim: 256
  output_dim: 128
  temperature: 0.07
  dropout: 0.1
  num_epochs: 50
  learning_rate: 0.001
  batch_size: 128
  evaluate_quality: true
```

### 4. 評価指標

- Accuracy, Precision, Recall, F1-score, AUC
- 5-fold Cross-Validation
- 統計的有意性検定（対応のあるt検定）
- 対照学習あり/なしの比較可視化

## 実行手順

### Step 1: 対照学習なし（ベースライン）

```bash
python src/experiments/run_named_models.py \
  --config config/robust_experiment.yaml \
  --experiment-id exp_25_1106_baseline \
  --only-model FreezedBertRgcnMlp \
  --best-rgcn-from-results data/training_results/exp_23_rgcn_FreezedBertRgcnMlp_sweep_named_models/experiment_results.json
```

```bash
python src/experiments/run_named_models.py \
  --config config/robust_experiment.yaml \
  --experiment-id exp_25_1106_baseline \
  --only-model FreezedBertMlp \
  --best-from-results data/training_results/exp_24_1105_FreezedBertMlp_named_models/experiment_results.json
```

```bash
python src/experiments/run_named_models.py \
  --config config/robust_experiment.yaml \
  --experiment-id exp_25_1106_baseline \
  --only-model FinetunedBertMlp \
  --best-from-results <FinetunedBertMlpのスイープ結果パス>
```

### Step 2: 対照学習あり（提案手法）

```bash
python src/experiments/run_named_models.py \
  --config config/robust_experiment.yaml \
  --experiment-id exp_25_1106_contrastive \
  --use-contrastive \
  --only-model FreezedBertRgcnMlp \
  --best-rgcn-from-results data/training_results/exp_23_rgcn_FreezedBertRgcnMlp_sweep_named_models/experiment_results.json
```

```bash
python src/experiments/run_named_models.py \
  --config config/robust_experiment.yaml \
  --experiment-id exp_25_1106_contrastive \
  --use-contrastive \
  --only-model FreezedBertMlp \
  --best-from-results data/training_results/exp_24_1105_FreezedBertMlp_named_models/experiment_results.json
```

```bash
python src/experiments/run_named_models.py \
  --config config/robust_experiment.yaml \
  --experiment-id exp_25_1106_contrastive \
  --use-contrastive \
  --only-model FinetunedBertMlp \
  --best-from-results <FinetunedBertMlpのスイープ結果パス>
```

## 期待される結果

### 仮説

1. **FreezedBertRgcnMlp**: 対照学習によりグラフ構造学習が改善される可能性が高い
2. **FreezedBertMlp**: 固定BERTなので対照学習の効果が大きい可能性
3. **FinetunedBertMlp**: BERTを微調整しているため、対照学習の効果は限定的かもしれない

### 分析項目

1. **性能改善度**: 各モデルでの対照学習による性能向上率
2. **統計的有意性**: 対応のあるt検定による有意差検定
3. **モデル間比較**: どのモデルが対照学習の恩恵を最も受けるか
4. **埋め込み品質**: 対照学習前後の埋め込み類似度分布の変化

## 出力ファイル

各実験IDごとに以下が生成される：

- `experiment_results.json`: 評価メトリクスと統計量
- `box_plots.png`: 箱ひげ図
- `bar_charts.png`: 棒グラフ
- `comprehensive_analysis.png`: 包括的分析
- `embedding_quality.json` (対照学習ありの場合): 埋め込み品質評価結果

## 注意事項

1. **再現性**: 同じシード（42）を使用
2. **計算リソース**: 対照学習は追加の計算時間が必要
3. **メモリ**: 対照学習ありの場合、メモリ使用量が増加する可能性
4. **比較の公平性**: 同じCV分割、同じネガティブサンプルを使用




