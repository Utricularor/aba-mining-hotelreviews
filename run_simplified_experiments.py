#!/usr/bin/env python3
"""Simplified experiments for ABA link prediction - focusing on what works."""

import os
import sys
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.aba_link_prediction.data_loaders import ABADataset
from src.aba_link_prediction.models import BERTLinkPredictor, CrossEncoderBERT
from src.aba_link_prediction.utils import set_seed, create_directories

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'data_path': 'data/output/Silver_Room_ContP_BodyN_4omini.csv',
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 16,
    'test_size': 0.2,
    'val_size': 0.1,
}

def train_simple_nn_model():
    """Train a simple neural network baseline."""
    logger.info("Training Simple NN baseline...")
    
    # Load datasets
    train_dataset = ABADataset(
        data_path=CONFIG['data_path'],
        mode='train',
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['seed']
    )
    
    val_dataset = ABADataset(
        data_path=CONFIG['data_path'],
        mode='val',
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['seed']
    )
    
    test_dataset = ABADataset(
        data_path=CONFIG['data_path'],
        mode='test',
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['seed']
    )
    
    # Get number of unique nodes for input dimension
    num_assumptions, num_propositions = train_dataset.get_num_nodes()
    input_dim = num_assumptions + num_propositions
    
    # Simple MLP model
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim):
            super(SimpleMLP, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
            self.assumption_emb = nn.Embedding(num_assumptions, input_dim)
            self.proposition_emb = nn.Embedding(num_propositions, input_dim)
            
        def forward(self, assumption_ids, proposition_ids):
            assumption_emb = self.assumption_emb(assumption_ids)
            proposition_emb = self.proposition_emb(proposition_ids)
            combined = torch.cat([assumption_emb, proposition_emb], dim=-1)
            return self.fc(combined).squeeze(-1)
    
    model = SimpleMLP(input_dim).to(CONFIG['device'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    num_epochs = 20
    train_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(batch['assumption_id'], batch['proposition_id'])
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                outputs = model(batch['assumption_id'], batch['proposition_id'])
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == batch['label']).sum().item()
                val_total += batch['label'].size(0)
        
        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: Loss = {train_losses[-1]:.4f}, Val Acc = {val_acc:.4f}")
    
    # Test evaluation
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            outputs = model(batch['assumption_id'], batch['proposition_id'])
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch['label'].cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)
    
    metrics = {
        'accuracy': accuracy_score(test_labels, test_preds),
        'precision': precision_score(test_labels, test_preds, zero_division=0),
        'recall': recall_score(test_labels, test_preds, zero_division=0),
        'f1': f1_score(test_labels, test_preds, zero_division=0),
        'roc_auc': roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
    }
    
    return metrics, train_losses, val_accs

def train_bert_model(model_type='bert', freeze_bert=True):
    """Train BERT-based models."""
    logger.info(f"Training {model_type} (freeze_bert={freeze_bert})...")
    
    # Load datasets
    train_dataset = ABADataset(
        data_path=CONFIG['data_path'],
        mode='train',
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['seed']
    )
    
    val_dataset = ABADataset(
        data_path=CONFIG['data_path'],
        mode='val',
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['seed']
    )
    
    test_dataset = ABADataset(
        data_path=CONFIG['data_path'],
        mode='test',
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['seed']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Initialize model
    if model_type == 'bert':
        model = BERTLinkPredictor(
            model_name='bert-base-uncased',
            hidden_dim=768,
            dropout=0.3,
            freeze_bert=freeze_bert,
            pooling_strategy='cls'
        )
    else:  # cross_encoder
        model = CrossEncoderBERT(
            model_name='bert-base-uncased',
            dropout=0.3,
            freeze_bert=freeze_bert
        )
    
    model = model.to(CONFIG['device'])
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=2e-5 if not freeze_bert else 1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    num_epochs = 3
    train_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(batch['assumption_text'], batch['proposition_text'])
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            epoch_correct += (preds == batch['label']).sum().item()
            epoch_total += batch['label'].size(0)
        
        train_losses.append(epoch_loss / len(train_loader))
        train_acc = epoch_correct / epoch_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(batch['assumption_text'], batch['proposition_text'])
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == batch['label']).sum().item()
                val_total += batch['label'].size(0)
        
        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        
        logger.info(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
    
    # Test evaluation
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(batch['assumption_text'], batch['proposition_text'])
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch['label'].cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)
    
    metrics = {
        'accuracy': accuracy_score(test_labels, test_preds),
        'precision': precision_score(test_labels, test_preds, zero_division=0),
        'recall': recall_score(test_labels, test_preds, zero_division=0),
        'f1': f1_score(test_labels, test_preds, zero_division=0),
        'roc_auc': roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
    }
    
    return metrics, train_losses, val_accs

def visualize_results(results):
    """Create visualizations of the results."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Bar plot of accuracy
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    ax = axes[0, 0]
    bars = ax.bar(range(len(models)), accuracies)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. F1 scores
    f1_scores = [results[m]['f1'] for m in models]
    
    ax = axes[0, 1]
    bars = ax.bar(range(len(models)), f1_scores)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('Model F1 Score Comparison')
    ax.set_ylim([0, 1])
    
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.3f}', ha='center', va='bottom')
    
    # 3. ROC-AUC scores
    roc_aucs = [results[m]['roc_auc'] for m in models]
    
    ax = axes[1, 0]
    bars = ax.bar(range(len(models)), roc_aucs)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Model ROC-AUC Comparison')
    ax.set_ylim([0, 1])
    
    for bar, auc in zip(bars, roc_aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.3f}', ha='center', va='bottom')
    
    # 4. Heatmap of all metrics
    metrics_data = []
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for model in models:
        metrics_data.append([results[model][m] for m in metric_names])
    
    ax = axes[1, 1]
    im = ax.imshow(metrics_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title('All Metrics Heatmap')
    
    # Add text annotations
    for i, model in enumerate(models):
        for j, metric in enumerate(metric_names):
            text = ax.text(j, i, f'{results[model][metric]:.2f}',
                          ha="center", va="center", color="black")
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('results/aba_link_prediction/model_comparison.png', dpi=100)
    logger.info("Saved visualization to results/aba_link_prediction/model_comparison.png")
    
    return fig

def main():
    """Run simplified experiments."""
    set_seed(CONFIG['seed'])
    create_directories(['results/aba_link_prediction'])
    
    results = {}
    
    # 1. Simple NN baseline
    logger.info("\n" + "="*50)
    logger.info("Experiment 1: Simple NN Baseline")
    metrics, train_losses, val_accs = train_simple_nn_model()
    results['Simple_NN'] = metrics
    logger.info(f"Results: {metrics}")
    
    # 2. BERT with frozen weights (fast)
    logger.info("\n" + "="*50)
    logger.info("Experiment 2: BERT (frozen)")
    metrics, train_losses, val_accs = train_bert_model('bert', freeze_bert=True)
    results['BERT_Frozen'] = metrics
    logger.info(f"Results: {metrics}")
    
    # 3. BERT with fine-tuning
    logger.info("\n" + "="*50)
    logger.info("Experiment 3: BERT (fine-tuned)")
    metrics, train_losses, val_accs = train_bert_model('bert', freeze_bert=False)
    results['BERT_Finetuned'] = metrics
    logger.info(f"Results: {metrics}")
    
    # 4. Cross-Encoder BERT (frozen)
    logger.info("\n" + "="*50)
    logger.info("Experiment 4: Cross-Encoder BERT (frozen)")
    metrics, train_losses, val_accs = train_bert_model('cross_encoder', freeze_bert=True)
    results['CrossEncoder_Frozen'] = metrics
    logger.info(f"Results: {metrics}")
    
    # 5. Cross-Encoder BERT (fine-tuned)
    logger.info("\n" + "="*50)
    logger.info("Experiment 5: Cross-Encoder BERT (fine-tuned)")
    metrics, train_losses, val_accs = train_bert_model('cross_encoder', freeze_bert=False)
    results['CrossEncoder_Finetuned'] = metrics
    logger.info(f"Results: {metrics}")
    
    # Save results
    with open('results/aba_link_prediction/simplified_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    visualize_results(results)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*50)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    logger.info(f"\nBest model (by F1-score): {best_model[0]} with F1={best_model[1]['f1']:.4f}")
    
    return results

if __name__ == '__main__':
    results = main()