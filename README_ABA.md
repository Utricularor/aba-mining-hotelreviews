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

## Citation

If you use this code, please cite the original ABA mining work and this implementation.