#!/usr/bin/env python3
"""Run experiments with balanced datasets using negative sampling."""

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.aba_link_prediction.data_loaders.balanced_dataset import BalancedABADataset, BalancedGraphDataset
from src.aba_link_prediction.models import RGCN, RGCNWithAttention, BERTLinkPredictor, CrossEncoderBERT
from src.aba_link_prediction.utils import set_seed, create_directories

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'data_path': 'data/output/Silver_Room_ContP_BodyN_4omini.csv',
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 32,
    'test_size': 0.2,
    'val_size': 0.1,
    'balance_ratio': 1.0,  # 1:1 ratio of negative to positive
    'sampling_strategy': 'undersample',  # undersample majority class
}

def train_simple_nn_balanced():
    """Train a simple neural network with balanced data."""
    logger.info("Training Simple NN with balanced dataset...")
    
    # Load balanced datasets
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
    
    # Get number of unique nodes
    num_assumptions, num_propositions = train_dataset.get_num_nodes()
    input_dim = 128  # Embedding dimension
    
    # Simple MLP model
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
            assumption_emb = self.assumption_emb(assumption_ids)
            proposition_emb = self.proposition_emb(proposition_ids)
            combined = torch.cat([assumption_emb, proposition_emb], dim=-1)
            return self.fc(combined).squeeze(-1)
    
    model = SimpleMLP().to(CONFIG['device'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Get class weights for balanced loss
    class_weights = train_dataset.get_class_weights().to(CONFIG['device'])
    
    # Training with weighted loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    
    num_epochs = 30
    best_val_f1 = 0
    train_losses = []
    val_metrics = []
    
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
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                outputs = model(batch['assumption_id'], batch['proposition_id'])
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch['label'].cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_probs = np.array(val_probs)
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}, Val F1 = {val_f1:.4f}")
    
    # Load best model
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
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    return metrics, cm, train_losses

def train_rgcn_balanced():
    """Train RGCN with balanced graph data."""
    logger.info("Training RGCN with balanced dataset...")
    
    # Load balanced graph dataset
    dataset = BalancedGraphDataset(
        data_path=CONFIG['data_path'],
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['seed'],
        balance_ratio=CONFIG['balance_ratio'],
        sampling_strategy=CONFIG['sampling_strategy']
    )
    
    graph_data = dataset.get_data()
    num_nodes = dataset.get_num_nodes()
    
    # Initialize RGCN model
    model = RGCN(
        num_nodes=num_nodes,
        hidden_dim=128,
        num_layers=2,
        num_relations=2,
        dropout=0.5,
        use_node_features=True
    ).to(CONFIG['device'])
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    num_epochs = 50
    best_val_f1 = 0
    train_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        x = dataset.get_node_features().to(CONFIG['device'])
        edge_index = graph_data['edge_index'].to(CONFIG['device'])
        edge_type = graph_data['edge_labels'].long().to(CONFIG['device'])
        
        train_edges = edge_index[:, graph_data['train_mask']]
        train_labels = graph_data['edge_labels'][graph_data['train_mask']].to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(x, edge_index, edge_type=edge_type, target_edges=train_edges)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_edges = edge_index[:, graph_data['val_mask']]
            val_labels = graph_data['edge_labels'][graph_data['val_mask']].to(CONFIG['device'])
            val_outputs = model(x, edge_index, edge_type=edge_type, target_edges=val_edges)
            val_probs = torch.sigmoid(val_outputs)
            val_preds = (val_probs > 0.5).float()
            
            val_acc = (val_preds == val_labels).float().mean().item()
            val_f1 = f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), zero_division=0)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}, Val Acc = {val_acc:.4f}, Val F1 = {val_f1:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_edges = edge_index[:, graph_data['test_mask']]
        test_labels = graph_data['edge_labels'][graph_data['test_mask']].cpu().numpy()
        test_outputs = model(x, edge_index, edge_type=edge_type, target_edges=test_edges)
        test_probs = torch.sigmoid(test_outputs).cpu().numpy()
        test_preds = (test_probs > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(test_labels, test_preds),
            'precision': precision_score(test_labels, test_preds, zero_division=0),
            'recall': recall_score(test_labels, test_preds, zero_division=0),
            'f1': f1_score(test_labels, test_preds, zero_division=0),
            'roc_auc': roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
        }
        
        cm = confusion_matrix(test_labels, test_preds)
    
    return metrics, cm, train_losses

def train_bert_balanced():
    """Train BERT with balanced data."""
    logger.info("Training BERT with balanced dataset...")
    
    # Load balanced datasets
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
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize BERT model
    model = BERTLinkPredictor(
        model_name='bert-base-uncased',
        hidden_dim=768,
        dropout=0.3,
        freeze_bert=True,  # Freeze for faster training
        pooling_strategy='cls'
    ).to(CONFIG['device'])
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    num_epochs = 5
    best_val_f1 = 0
    train_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(batch['assumption_text'], batch['proposition_text'])
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(CONFIG['device']) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(batch['assumption_text'], batch['proposition_text'])
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch['label'].cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
        
        logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val F1 = {val_f1:.4f}")
    
    # Load best model
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
    
    cm = confusion_matrix(test_labels, test_preds)
    
    return metrics, cm, train_losses

def plot_results(results):
    """Create comprehensive visualization of results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Metrics comparison
    models = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.15
    
    for i, metric in enumerate(metrics_names):
        values = [results[m]['metrics'][metric] for m in models]
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2-4. Confusion matrices
    for idx, (model_name, data) in enumerate(results.items()):
        ax = axes[0, idx + 1] if idx < 2 else axes[1, idx - 2]
        cm = data['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax.set_title(f'{model_name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # 5. Training loss curves
    ax = axes[1, 2]
    for model_name, data in results.items():
        if 'train_losses' in data:
            ax.plot(data['train_losses'], label=model_name)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/aba_link_prediction/balanced_results.png', dpi=100)
    logger.info("Saved visualization to results/aba_link_prediction/balanced_results.png")
    
    return fig

def main():
    """Run all balanced experiments."""
    set_seed(CONFIG['seed'])
    create_directories(['results/aba_link_prediction'])
    
    results = {}
    
    # 1. Simple NN with balanced data
    logger.info("\n" + "="*50)
    logger.info("Experiment 1: Simple NN (Balanced)")
    metrics, cm, losses = train_simple_nn_balanced()
    results['Simple_NN_Balanced'] = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'train_losses': losses
    }
    logger.info(f"Results: {metrics}")
    
    # 2. RGCN with balanced data
    logger.info("\n" + "="*50)
    logger.info("Experiment 2: RGCN (Balanced)")
    metrics, cm, losses = train_rgcn_balanced()
    results['RGCN_Balanced'] = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'train_losses': losses
    }
    logger.info(f"Results: {metrics}")
    
    # 3. BERT with balanced data
    logger.info("\n" + "="*50)
    logger.info("Experiment 3: BERT (Balanced)")
    metrics, cm, losses = train_bert_balanced()
    results['BERT_Balanced'] = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'train_losses': losses
    }
    logger.info(f"Results: {metrics}")
    
    # Save results
    results_for_json = {}
    for model_name, data in results.items():
        results_for_json[model_name] = {
            'metrics': data['metrics'],
            'confusion_matrix': data['confusion_matrix'].tolist()
        }
    
    with open('results/aba_link_prediction/balanced_results.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    # Visualize results
    plot_results(results)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("BALANCED EXPERIMENT SUMMARY")
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
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['f1'])
    logger.info(f"\nBest model (by F1-score): {best_model[0]} with F1={best_model[1]['metrics']['f1']:.4f}")
    
    return results

if __name__ == '__main__':
    results = main()