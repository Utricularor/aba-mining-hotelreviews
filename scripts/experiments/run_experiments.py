#!/usr/bin/env python3
"""Run all experiments for ABA link prediction."""

import os
import sys
import json
import time
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.aba_link_prediction.data_loaders import ABADataset, ABAGraphDataset
from src.aba_link_prediction.models import RGCN, RGCNWithAttention, BERTLinkPredictor, CrossEncoderBERT
from src.aba_link_prediction.models.contrastive import GraphContrastiveLearning
from src.aba_link_prediction.utils import set_seed, create_directories

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'data_path': '../../data/output/Silver_Room_ContP_BodyN_4omini.csv',
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_epochs': 20,  # Reduced for faster experiments
    'learning_rate': 0.001,
    'batch_size': 32,
    'test_size': 0.2,
    'val_size': 0.1,
    'contrastive_epochs': 10,  # Reduced for faster experiments
}

def train_graph_model(model_type='rgcn', use_contrastive=False):
    """Train graph-based models."""
    logger.info(f"Training {model_type} (contrastive={use_contrastive})...")
    
    # Load data
    dataset = ABAGraphDataset(
        data_path=CONFIG['data_path'],
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['seed']
    )
    
    graph_data = dataset.get_data()
    num_nodes = dataset.get_num_nodes()
    
    # Initialize model
    if model_type == 'rgcn':
        model = RGCN(
            num_nodes=num_nodes,
            hidden_dim=64,  # Reduced for faster training
            num_layers=2,
            num_relations=2,
            dropout=0.5,
            use_node_features=True
        )
    else:  # rgcn_attention
        model = RGCNWithAttention(
            num_nodes=num_nodes,
            hidden_dim=64,
            num_layers=2,
            num_relations=2,
            num_heads=4,
            dropout=0.5,
            use_node_features=True
        )
    
    model = model.to(CONFIG['device'])
    
    # Contrastive pre-training
    if use_contrastive:
        logger.info("Starting contrastive pre-training...")
        contrastive_model = GraphContrastiveLearning(
            encoder=model,
            temperature=0.07,
            augmentation_prob=0.2
        )
        
        optimizer = optim.Adam(contrastive_model.parameters(), lr=0.0003)
        
        for epoch in range(CONFIG['contrastive_epochs']):
            x = dataset.get_node_features().to(CONFIG['device'])
            edge_index = graph_data['edge_index'].to(CONFIG['device'])
            
            z1, z2 = contrastive_model(x, edge_index)
            loss = contrastive_model.compute_loss(z1, z2, batch_size=128)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                logger.info(f"Contrastive Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Main training
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    train_losses = []
    val_accs = []
    
    for epoch in range(CONFIG['num_epochs']):
        # Training
        model.train()
        x = dataset.get_node_features().to(CONFIG['device'])
        edge_index = graph_data['edge_index'].to(CONFIG['device'])
        # Create edge types (0 for non-contrary, 1 for contrary)
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
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            val_acc = (val_preds == val_labels).float().mean().item()
            val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}, Val Acc = {val_acc:.4f}")
    
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
    
    return metrics, train_losses, val_accs

def train_bert_model(model_type='bert'):
    """Train BERT-based models."""
    logger.info(f"Training {model_type}...")
    
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
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    if model_type == 'bert':
        model = BERTLinkPredictor(
            model_name='bert-base-uncased',
            hidden_dim=768,
            dropout=0.3,
            freeze_bert=True,  # Freeze for faster training
            pooling_strategy='cls'
        )
    else:  # cross_encoder
        model = CrossEncoderBERT(
            model_name='bert-base-uncased',
            dropout=0.3,
            freeze_bert=True  # Freeze for faster training
        )
    
    model = model.to(CONFIG['device'])
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.00002)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    train_losses = []
    val_accs = []
    
    # Reduced epochs for BERT
    num_epochs = 5 if model_type == 'bert' else 3
    
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
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

def main():
    """Run all experiments."""
    set_seed(CONFIG['seed'])
    create_directories(['results/aba_link_prediction'])
    
    results = {}
    
    # 1. RGCN without contrastive learning
    logger.info("\n" + "="*50)
    logger.info("Experiment 1: RGCN without contrastive learning")
    metrics, train_losses, val_accs = train_graph_model('rgcn', use_contrastive=False)
    results['RGCN'] = metrics
    logger.info(f"Results: {metrics}")
    
    # 2. RGCN with contrastive learning
    logger.info("\n" + "="*50)
    logger.info("Experiment 2: RGCN with contrastive learning")
    metrics, train_losses, val_accs = train_graph_model('rgcn', use_contrastive=True)
    results['RGCN_Contrastive'] = metrics
    logger.info(f"Results: {metrics}")
    
    # 3. RGCN-Attention without contrastive learning
    logger.info("\n" + "="*50)
    logger.info("Experiment 3: RGCN-Attention without contrastive learning")
    metrics, train_losses, val_accs = train_graph_model('rgcn_attention', use_contrastive=False)
    results['RGCN_Attention'] = metrics
    logger.info(f"Results: {metrics}")
    
    # 4. RGCN-Attention with contrastive learning
    logger.info("\n" + "="*50)
    logger.info("Experiment 4: RGCN-Attention with contrastive learning")
    metrics, train_losses, val_accs = train_graph_model('rgcn_attention', use_contrastive=True)
    results['RGCN_Attention_Contrastive'] = metrics
    logger.info(f"Results: {metrics}")
    
    # 5. BERT
    logger.info("\n" + "="*50)
    logger.info("Experiment 5: BERT Bi-encoder")
    metrics, train_losses, val_accs = train_bert_model('bert')
    results['BERT'] = metrics
    logger.info(f"Results: {metrics}")
    
    # 6. Cross-Encoder BERT
    logger.info("\n" + "="*50)
    logger.info("Experiment 6: Cross-Encoder BERT")
    metrics, train_losses, val_accs = train_bert_model('cross_encoder')
    results['Cross_Encoder_BERT'] = metrics
    logger.info(f"Results: {metrics}")
    
    # Save results
    with open('results/aba_link_prediction/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
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