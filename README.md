# ABA Link Prediction for Hotel Reviews

This module implements various models for predicting attack links in Argumentation-Based Analysis (ABA) using hotel review data.

## Features

- **RGCN Models**: Relational Graph Convolutional Networks with optional attention mechanisms
- **BERT Models**: Bi-encoder and cross-encoder BERT architectures for text-based prediction
- **Contrastive Learning**: Pre-training methods for improving node representations
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1, ROC-AUC, and graph-specific metrics

## Installation

```bash
pip install -r requirements_aba.txt
```

## Quick Start

### 1. Train RGCN Model

```bash
# Basic RGCN
python train_rgcn.py --model rgcn

# RGCN with attention
python train_rgcn.py --model rgcn_attention

# With contrastive pre-training
python train_rgcn.py --model rgcn --pretrain
```

### 2. Train BERT Model

```bash
# Bi-encoder BERT
python train_bert.py --model bert

# Cross-encoder BERT
python train_bert.py --model cross_encoder
```

### 3. Compare Models

```bash
python compare_models.py
```

## Project Structure

```
src/aba_link_prediction/
├── config/           # Configuration files
├── data_loaders/     # Data loading utilities
├── models/           # Model implementations
│   ├── rgcn.py      # RGCN models
│   ├── bert_classifier.py  # BERT models
│   └── contrastive.py      # Contrastive learning
├── trainers/         # Training utilities
├── evaluators/       # Evaluation metrics
└── utils/           # Helper functions
```

## Configuration

Edit `src/aba_link_prediction/config/config.yaml` to customize:
- Data paths and preprocessing
- Model hyperparameters
- Training settings
- Evaluation metrics

## Models

### RGCN (Relational Graph Convolutional Network)
- Captures graph structure of argumentation
- Supports multiple relation types
- Optional attention mechanism

### BERT-based Models
- **Bi-encoder**: Encodes assumptions and propositions separately
- **Cross-encoder**: Jointly encodes assumption-proposition pairs
- Supports various pooling strategies

### Contrastive Learning
- SimCLR-based pre-training
- Graph contrastive learning with augmentation
- Improves node representation quality

## Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score
- **Ranking**: ROC-AUC, Average Precision
- **Graph-specific**: Hits@K, Mean Reciprocal Rank (MRR)

## Results

Results are saved in:
- `models/aba_link_prediction/`: Trained model checkpoints
- `results/aba_link_prediction/`: Evaluation metrics and plots
- `logs/aba_link_prediction/`: Training logs

## Usage Example

```python
from src.aba_link_prediction.data_loaders import ABADataset
from src.aba_link_prediction.models import BERTLinkPredictor
from src.aba_link_prediction.trainers import BERTTrainer

# Load data
dataset = ABADataset('data/output/Silver_Room_ContP_BodyN_4omini.csv')

# Initialize model
model = BERTLinkPredictor(model_name='bert-base-uncased')

# Train
trainer = BERTTrainer(model)
trainer.train(train_loader, val_loader, num_epochs=10)
```

## Running Robust Experiments

### 実験の実行

設定ファイルを編集して実験をカスタマイズできます：

```bash
# デフォルト設定で実行（実験IDは自動生成されます）
python src/experiments/run_robust_experiment.py

# 実験IDを指定して実行
python src/experiments/run_robust_experiment.py --experiment-id exp001

# カスタム設定ファイルと実験IDを指定
python src/experiments/run_robust_experiment.py --config my_config.yaml --experiment-id exp002
```

### 実験IDの指定方法

実験IDは3つの方法で指定できます（優先順位順）：

1. **コマンドライン引数**: `--experiment-id <ID>`
2. **YAML設定ファイル**: `data.experiment_id: "exp001"`
3. **自動生成**: どちらも指定しない場合、タイムスタンプで自動生成（例: `exp_20250107_143022`）

### 設定ファイルの主要項目

- `data`: データファイルのパスと出力先
  - `base_output_dir`: 結果の保存先ベースディレクトリ
  - `experiment_id`: 実験ID（nullの場合は自動生成）
- `negative_sampling`: ネガティブサンプリング戦略の設定
- `cross_validation`: Cross-validationの設定
- `models`: 各モデルの有効化とハイパーパラメータ
- `visualization`: 可視化の設定

### 実験結果

実験結果は実験IDごとにディレクトリが作成され、以下に保存されます：

- `data/training_results/<実験ID>/experiment_results.json`: 評価メトリクスと統計量
- `data/training_results/<実験ID>/*.png`: 可視化グラフ（設定で有効化した場合）

例：
```
data/training_results/
├── exp001/
│   ├── experiment_results.json
│   ├── box_plots.png
│   └── bar_charts.png
├── exp002/
│   ├── experiment_results.json
│   └── ...
└── exp_20250107_143022/  # 自動生成された実験ID
    └── ...
```

## Testing

### テスト実行手順

#### すべてのテストを実行

```bash
# プロジェクトルートで実行
pytest tests/
```

#### ユニットテストのみ実行

```bash
pytest tests/unit/
```

#### 統合テストのみ実行

```bash
pytest tests/integration/
```

#### カバレッジレポート付きで実行

```bash
pytest --cov=src --cov-report=html tests/
```

#### 特定のテストファイルを実行

```bash
# モデルのテスト
pytest tests/unit/test_models.py

# Cross-validationのテスト
pytest tests/unit/test_cross_validation.py

# 可視化のテスト
pytest tests/unit/test_visualization.py

# パイプライン全体のテスト
pytest tests/integration/test_experiment_pipeline.py
```

#### 詳細な出力で実行

```bash
pytest -v tests/
```

### テスト構成

- `tests/conftest.py`: pytest設定とフィクスチャ
- `tests/unit/`: ユニットテスト（個別の関数・クラスのテスト）
  - `test_models.py`: モデル定義のテスト
  - `test_cross_validation.py`: Cross-validation機能のテスト
  - `test_visualization.py`: 可視化・統計分析のテスト
- `tests/integration/`: 統合テスト（エンドツーエンドのテスト）
  - `test_experiment_pipeline.py`: 実験パイプライン全体のテスト

### モックデータについて

テストはすべてモックデータを使用して実行されます。実データは不要です。
モックデータは `tests/conftest.py` で定義されています。

## Project Structure (Updated)

```
.
├── config/
│   └── robust_experiment.yaml       # 実験設定ファイル
├── src/
│   ├── preprocess/                  # データ前処理
│   ├── augmentation/                # データ拡張（ネガティブサンプリング）
│   ├── model_defs/                  # モデル定義
│   │   └── models.py               # R-GCN, BERT models, Baselines
│   ├── model_training/              # 学習・評価
│   │   ├── train.py                # R-GCN学習
│   │   ├── train_bert.py           # BERT学習
│   │   └── evaluate.py             # 評価関数
│   ├── experiments/                 # 実験実行
│   │   ├── cross_validation.py     # Cross-validation
│   │   └── run_robust_experiment.py # メイン実験スクリプト
│   └── visualization/               # 可視化・統計分析
│       └── plot_results.py
├── tests/
│   ├── conftest.py                 # pytest設定
│   ├── unit/                       # ユニットテスト
│   │   ├── test_models.py
│   │   ├── test_cross_validation.py
│   │   └── test_visualization.py
│   └── integration/                # 統合テスト
│       └── test_experiment_pipeline.py
├── data/
│   ├── output/                     # 入力データ
│   └── training_results/           # 実験結果
└── README.md
```

## Citation

If you use this code, please cite the original ABA mining work and this implementation.