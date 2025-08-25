# ABA Link Prediction - Comprehensive Experiment Report

Generated: 2025-08-16 20:41:26

## Executive Summary

- **Best F1 Score**: 0.0825 (Simple NN - Robust (Balanced))
- **Best ROC-AUC**: 0.8726 (Simple NN - Robust (Balanced))
- **Total Experiments Run**: 3
- **Number of Unique Models**: 3

## Key Findings

2. **Best Model Architecture**: Simple NN achieved the highest average F1 score (0.0825)

## Detailed Results by Experiment

### Robust (Balanced)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Simple NN | 0.7465 | 0.0433 | 0.8706 | 0.0825 | 0.8726 |
| BERT Bi-Encoder | 0.7837 | 0.0397 | 0.6706 | 0.0750 | 0.7848 |
| BERT Cross-Encoder | 0.4895 | 0.0184 | 0.7255 | 0.0359 | 0.6694 |

## Model Architecture Comparison

| Model | Mean F1 | Std F1 | Max F1 | Mean AUC | Std AUC | Max AUC | Mean Acc |
|-------|---------|--------|--------|----------|---------|---------|----------|
| BERT Bi-Encoder | 0.0750 | nan | 0.0750 | 0.7848 | nan | 0.7848 | 0.7837 |
| BERT Cross-Encoder | 0.0359 | nan | 0.0359 | 0.6694 | nan | 0.6694 | 0.4895 |
| Simple NN | 0.0825 | nan | 0.0825 | 0.8726 | nan | 0.8726 | 0.7465 |

## Recommendations

Based on the comprehensive experiments, we recommend:

1. **For Production Use**: Simple NN with Robust (Balanced) approach (F1: 0.0825)
2. **For High Recall Requirements**: Consider models with balanced datasets
3. **For Computational Efficiency**: Simple NN models provide competitive performance with faster training times

## Technical Details

- **Dataset**: Silver_Room_ContP_BodyN_4omini.csv
- **Class Distribution**: Originally 1.31% positive, 98.69% negative
- **Balancing Strategies**: Undersampling, Hard Negative Sampling
- **Evaluation**: 5-fold cross-validation, held-out test set
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC