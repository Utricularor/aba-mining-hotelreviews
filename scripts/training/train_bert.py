#!/usr/bin/env python3
"""Train BERT model for ABA link prediction."""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.aba_link_prediction.data_loaders import ABADataset
from src.aba_link_prediction.models import BERTLinkPredictor, CrossEncoderBERT
from src.aba_link_prediction.trainers import BERTTrainer
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
    
    # Load datasets
    logging.info(f"Loading data from {config['data']['train_path']}")
    
    train_dataset = ABADataset(
        data_path=config['data']['train_path'],
        mode='train',
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    val_dataset = ABADataset(
        data_path=config['data']['train_path'],
        mode='val',
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    test_dataset = ABADataset(
        data_path=config['data']['train_path'],
        mode='test',
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Initialize model
    if args.model == 'bert':
        model_config = config['models']['bert']
        model = BERTLinkPredictor(
            model_name=model_config['model_name'],
            hidden_dim=model_config['hidden_dim'],
            dropout=model_config['dropout'],
            freeze_bert=model_config['freeze_bert'],
            pooling_strategy=model_config['pooling_strategy']
        )
    elif args.model == 'cross_encoder':
        model_config = config['models']['cross_encoder']
        model = CrossEncoderBERT(
            model_name=model_config['model_name'],
            dropout=model_config['dropout'],
            freeze_bert=model_config['freeze_bert']
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    logging.info(f"Model: {args.model}")
    logging.info(f"Number of parameters: {count_parameters(model):,}")
    
    # Setup trainer
    trainer = BERTTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['bert_learning_rate'],
        weight_decay=config['training']['weight_decay'],
        scheduler_type=config['training']['scheduler_type'],
        warmup_steps=config['training']['warmup_steps']
    )
    
    # Train model
    logging.info("Starting training...")
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['bert_num_epochs'],
        save_dir=Path(config['paths']['models_dir']),
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=Path(config['paths']['results_dir']) / f'{args.model}_training_history.png'
    )
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    evaluator = ModelEvaluator(model, device)
    test_metrics = evaluator.evaluate_dataset(test_loader, model_type='bert')
    evaluator.print_evaluation_report(test_metrics, dataset_name='Test')
    
    # Save final model
    save_model(
        model,
        Path(config['paths']['models_dir']) / f'{args.model}_final.pt',
        metrics=test_metrics
    )
    
    logging.info("Training completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT model for ABA link prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='src/aba_link_prediction/config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='bert',
        choices=['bert', 'cross_encoder'],
        help='Model type to train'
    )
    
    args = parser.parse_args()
    main(args)