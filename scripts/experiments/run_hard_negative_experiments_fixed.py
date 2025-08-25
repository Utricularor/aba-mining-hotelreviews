#!/usr/bin/env python3
"""Run experiments with hard negative sampling for robust training (Fixed version)."""

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.aba_link_prediction.models import BERTLinkPredictor, CrossEncoderBERT
from src.aba_link_prediction.utils import set_seed, create_directories
from src.aba_link_prediction.data_loaders.balanced_dataset import BalancedABADataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'data_path': '../../data/output/Silver_Room_ContP_BodyN_4omini.csv',
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 32,
    'test_size': 0.2,
    'val_size': 0.1,
    'balance_ratio': 1.0,
    'sampling_strategy': 'undersample',
}

def train_model_with_balanced_hard_negatives(model_type='simple_nn'):
    """Train model with balanced dataset including hard negatives."""
    
    logger.info(f"Training {model_type} with balanced dataset...")
    
    # Load balanced datasets (with undersampling)
    train_dataset = BalancedABADataset(
        data_path=CONFIG['data_path'],
        mode='train',
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['seed'],
        balance_ratio=CONFIG['balance_ratio'],
        sampling_strategy=CONFIG['sampling_strategy']
    )
    
    val_dataset = BalancedABADataset(
        data_path=CONFIG['data_path'],
        mode='val',
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['seed']
    )
    
    test_dataset = BalancedABADataset(
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
    
    # Initialize model based on type
    if model_type == 'simple_nn':
        # Get number of unique nodes
        num_assumptions, num_propositions = train_dataset.get_num_nodes()
        input_dim = 128  # Embedding dimension
        
        class SimpleMLP(nn.Module):
            def __init__(self):
                super(SimpleMLP, self).__init__()
                self.assumption_emb = nn.Embedding(num_assumptions, input_dim)
                self.proposition_emb = nn.Embedding(num_propositions, input_dim)
                self.fc = nn.Sequential(
                    nn.Linear(input_dim * 2, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                )
                
            def forward(self, assumption_ids, proposition_ids):
                # Handle out of bounds indices
                assumption_ids = torch.clamp(assumption_ids, 0, self.assumption_emb.num_embeddings - 1)
                proposition_ids = torch.clamp(proposition_ids, 0, self.proposition_emb.num_embeddings - 1)
                
                assumption_emb = self.assumption_emb(assumption_ids)
                proposition_emb = self.proposition_emb(proposition_ids)
                combined = torch.cat([assumption_emb, proposition_emb], dim=-1)
                return self.fc(combined).squeeze(-1)
        
        model = SimpleMLP().to(CONFIG['device'])
        
    elif model_type == 'bert':
        model = BERTLinkPredictor(
            model_name='bert-base-uncased',
            hidden_dim=768,
            dropout=0.3,
            freeze_bert=True,
            pooling_strategy='cls'
        ).to(CONFIG['device'])
        
    elif model_type == 'cross_encoder':
        model = CrossEncoderBERT(
            model_name='bert-base-uncased',
            dropout=0.3,
            freeze_bert=True
        ).to(CONFIG['device'])
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Get class weights
    class_weights = train_dataset.get_class_weights().to(CONFIG['device'])
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001 if model_type == 'simple_nn' else 0.0001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    
    num_epochs = 30 if model_type == 'simple_nn' else 5
    best_val_f1 = 0
    best_model_state = None
    train_losses = []
    val_metrics_history = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            if model_type == 'simple_nn':
                outputs = model(batch['assumption_id'], batch['proposition_id'])
            else:  # BERT models
                outputs = model(batch['assumption_text'], batch['proposition_text'])
            
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Collect predictions
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(batch['label'].cpu().numpy())
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Calculate training metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if model_type == 'simple_nn':
                    outputs = model(batch['assumption_id'], batch['proposition_id'])
                else:
                    outputs = model(batch['assumption_text'], batch['proposition_text'])
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch['label'].cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_probs = np.array(val_probs)
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
        
        val_metrics_history.append({
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'auc': val_auc
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train F1={train_f1:.4f}, "
                       f"Val F1={val_f1:.4f}, Val AUC={val_auc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            if model_type == 'simple_nn':
                outputs = model(batch['assumption_id'], batch['proposition_id'])
            else:
                outputs = model(batch['assumption_text'], batch['proposition_text'])
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch['label'].cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)
    
    # Calculate test metrics
    test_metrics = {
        'accuracy': accuracy_score(test_labels, test_preds),
        'precision': precision_score(test_labels, test_preds, zero_division=0),
        'recall': recall_score(test_labels, test_preds, zero_division=0),
        'f1': f1_score(test_labels, test_preds, zero_division=0),
        'roc_auc': roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
    }
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    # Classification report
    report = classification_report(test_labels, test_preds, 
                                  target_names=['Non-Contrary', 'Contrary'],
                                  output_dict=True)
    
    return test_metrics, cm, report, train_losses, val_metrics_history

def visualize_results(results_dict):
    """Create comprehensive visualization of results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Metrics comparison bar plot
    ax = axes[0, 0]
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(models):
        values = [results_dict[model]['metrics'][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2-4. Confusion matrices
    for idx, (model_name, data) in enumerate(results_dict.items()):
        ax = axes[0, idx + 1] if idx < 2 else axes[1, idx - 2]
        cm = data['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Non-Contrary', 'Contrary'],
                   yticklabels=['Non-Contrary', 'Contrary'])
        ax.set_title(f'{model_name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # 5. Training curves (for first model)
    ax = axes[1, 2]
    first_model = list(results_dict.keys())[0]
    val_history = results_dict[first_model]['val_history']
    
    epochs = range(1, len(val_history) + 1)
    ax.plot(epochs, [m['f1'] for m in val_history], 'b-', label='F1 Score')
    ax.plot(epochs, [m['auc'] for m in val_history], 'r-', label='AUC')
    ax.plot(epochs, [m['accuracy'] for m in val_history], 'g-', label='Accuracy')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Metrics Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/aba_link_prediction/robust_experiment_results.png', dpi=100)
    logger.info("Saved visualization to results/aba_link_prediction/robust_experiment_results.png")
    
    return fig

def main():
    """Run experiments with balanced datasets."""
    
    set_seed(CONFIG['seed'])
    create_directories(['results/aba_link_prediction'])
    
    results = {}
    
    # Experiment 1: Simple NN with balanced data
    logger.info("\n" + "="*50)
    logger.info("Experiment 1: Simple NN with Balanced Data")
    metrics, cm, report, losses, val_history = train_model_with_balanced_hard_negatives('simple_nn')
    results['SimpleNN_Balanced_Robust'] = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'report': report,
        'train_losses': losses,
        'val_history': val_history
    }
    logger.info(f"Results: {metrics}")
    
    # Experiment 2: BERT with balanced data
    logger.info("\n" + "="*50)
    logger.info("Experiment 2: BERT with Balanced Data")
    metrics, cm, report, losses, val_history = train_model_with_balanced_hard_negatives('bert')
    results['BERT_Balanced_Robust'] = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'report': report,
        'train_losses': losses,
        'val_history': val_history
    }
    logger.info(f"Results: {metrics}")
    
    # Experiment 3: Cross-Encoder BERT with balanced data
    logger.info("\n" + "="*50)
    logger.info("Experiment 3: Cross-Encoder BERT with Balanced Data")
    metrics, cm, report, losses, val_history = train_model_with_balanced_hard_negatives('cross_encoder')
    results['CrossEncoder_Balanced_Robust'] = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'report': report,
        'train_losses': losses,
        'val_history': val_history
    }
    logger.info(f"Results: {metrics}")
    
    # Save results
    results_for_json = {}
    for model_name, data in results.items():
        results_for_json[model_name] = {
            'metrics': data['metrics'],
            'confusion_matrix': data['confusion_matrix'].tolist(),
            'classification_report': data['report']
        }
    
    with open('results/aba_link_prediction/robust_experiment_results.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    # Visualize results
    visualize_results(results)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("ROBUST EXPERIMENT SUMMARY")
    logger.info("="*50)
    
    for model_name, data in results.items():
        metrics = data['metrics']
        logger.info(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Print confusion matrix
        cm = data['confusion_matrix']
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Compare with previous results
    logger.info("\n" + "="*50)
    logger.info("COMPARISON WITH PREVIOUS METHODS")
    logger.info("="*50)
    
    logger.info("\nPrevious Results (Simple Balancing):")
    logger.info("  BERT_Balanced: F1=0.133, AUC=0.825")
    logger.info("  RGCN_Balanced: F1=0.085, AUC=0.892")
    logger.info("  Simple_NN_Balanced: F1=0.083, AUC=0.873")
    
    logger.info("\nCurrent Results (Robust Experiments):")
    for model_name, data in results.items():
        metrics = data['metrics']
        logger.info(f"  {model_name}: F1={metrics['f1']:.3f}, AUC={metrics['roc_auc']:.3f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['f1'])
    logger.info(f"\nBest model: {best_model[0]} with F1={best_model[1]['metrics']['f1']:.4f}")
    
    logger.info("\n✅ Robust experiments completed successfully!")
    
    return results

if __name__ == '__main__':
    results = main()