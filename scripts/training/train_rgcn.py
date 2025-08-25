#!/usr/bin/env python3
"""Train RGCN model for ABA link prediction."""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.aba_link_prediction.data_loaders import ABAGraphDataset
from src.aba_link_prediction.models import RGCN, RGCNWithAttention
from src.aba_link_prediction.models.contrastive import GraphContrastiveLearning
from src.aba_link_prediction.trainers import RGCNTrainer, ContrastiveTrainer
from src.aba_link_prediction.evaluators import ModelEvaluator
from src.aba_link_prediction.utils import (
    set_seed, load_config, setup_logging, create_directories,
    save_model, plot_training_history, get_device, count_parameters
)


def main(args):
    """Main training function."""
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    set_seed(config['experiment']['seed'])
    setup_logging(
        log_file=config['logging']['file'],
        level=config['logging']['level']
    )
    
    # Create directories
    create_directories([
        config['paths']['models_dir'],
        config['paths']['logs_dir'],
        config['paths']['results_dir']
    ])
    
    # Get device
    device = get_device(config['experiment']['device'] == 'cuda')
    
    # Load data
    logging.info(f"Loading data from {config['data']['train_path']}")
    dataset = ABAGraphDataset(
        data_path=config['data']['train_path'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    graph_data = dataset.get_data()
    num_nodes = dataset.get_num_nodes()
    
    # Initialize model
    if args.model == 'rgcn':
        model_config = config['models']['rgcn']
        model = RGCN(
            num_nodes=num_nodes,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_relations=model_config['num_relations'],
            dropout=model_config['dropout'],
            use_node_features=model_config['use_node_features']
        )
    elif args.model == 'rgcn_attention':
        model_config = config['models']['rgcn_attention']
        model = RGCNWithAttention(
            num_nodes=num_nodes,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_relations=model_config['num_relations'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout'],
            use_node_features=model_config['use_node_features']
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    logging.info(f"Model: {args.model}")
    logging.info(f"Number of parameters: {count_parameters(model):,}")
    
    # Contrastive pre-training if specified
    if args.pretrain:
        logging.info("Starting contrastive pre-training...")
        
        contrastive_model = GraphContrastiveLearning(
            encoder=model,
            temperature=config['training']['contrastive']['temperature'],
            augmentation_prob=config['training']['contrastive']['augmentation_prob']
        )
        
        contrastive_trainer = ContrastiveTrainer(
            model=contrastive_model,
            device=device,
            learning_rate=config['training']['contrastive']['learning_rate']
        )
        
        # Create dummy dataloader for contrastive learning
        # In practice, you'd want a proper dataloader here
        contrastive_data = [{
            'x': dataset.get_node_features(),
            'edge_index': graph_data['edge_index'],
            'edge_type': None
        }]
        
        contrastive_loader = DataLoader(
            contrastive_data,
            batch_size=1,
            shuffle=True
        )
        
        contrastive_history = contrastive_trainer.train(
            contrastive_loader,
            num_epochs=config['training']['contrastive']['num_epochs'],
            save_dir=Path(config['paths']['models_dir'])
        )
        
        logging.info("Contrastive pre-training completed")
    
    # Setup trainer
    trainer = RGCNTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        scheduler_type=config['training']['scheduler_type'],
        warmup_steps=config['training']['warmup_steps']
    )
    
    # Create data loaders
    # For graph models, we typically work with the full graph
    # Here we create pseudo-batches for training
    train_data = [{
        'edge_index': graph_data['edge_index'][:, graph_data['train_mask']],
        'label': graph_data['edge_labels'][graph_data['train_mask']],
        'x': dataset.get_node_features(),
        'target_edges': graph_data['edge_index'][:, graph_data['train_mask']]
    }]
    
    val_data = [{
        'edge_index': graph_data['edge_index'][:, graph_data['train_mask']],
        'label': graph_data['edge_labels'][graph_data['val_mask']],
        'x': dataset.get_node_features(),
        'target_edges': graph_data['edge_index'][:, graph_data['val_mask']]
    }]
    
    train_loader = DataLoader(train_data, batch_size=1)
    val_loader = DataLoader(val_data, batch_size=1)
    
    # Train model
    logging.info("Starting training...")
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs'],
        save_dir=Path(config['paths']['models_dir']),
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=Path(config['paths']['results_dir']) / 'training_history.png'
    )
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_data = [{
        'edge_index': graph_data['edge_index'][:, graph_data['train_mask']],
        'label': graph_data['edge_labels'][graph_data['test_mask']],
        'x': dataset.get_node_features(),
        'target_edges': graph_data['edge_index'][:, graph_data['test_mask']]
    }]
    
    test_loader = DataLoader(test_data, batch_size=1)
    
    evaluator = ModelEvaluator(model, device)
    test_metrics = evaluator.evaluate_dataset(test_loader, model_type='rgcn')
    evaluator.print_evaluation_report(test_metrics, dataset_name='Test')
    
    # Save final model
    save_model(
        model,
        Path(config['paths']['models_dir']) / 'final_model.pt',
        metrics=test_metrics
    )
    
    logging.info("Training completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RGCN model for ABA link prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='src/aba_link_prediction/config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='rgcn',
        choices=['rgcn', 'rgcn_attention'],
        help='Model type to train'
    )
    parser.add_argument(
        '--pretrain',
        action='store_true',
        help='Use contrastive pre-training'
    )
    
    args = parser.parse_args()
    main(args)