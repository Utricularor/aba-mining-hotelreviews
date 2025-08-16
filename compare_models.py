#!/usr/bin/env python3
"""Compare different models for ABA link prediction."""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.aba_link_prediction.data_loaders import ABADataset, ABAGraphDataset
from src.aba_link_prediction.models import (
    RGCN, RGCNWithAttention, BERTLinkPredictor, CrossEncoderBERT
)
from src.aba_link_prediction.evaluators import compare_models
from src.aba_link_prediction.utils import (
    set_seed, load_config, setup_logging, create_directories,
    load_model, save_results, get_device
)


def load_trained_model(model_class, model_path, config, device, **kwargs):
    """Load a trained model from checkpoint."""
    model = model_class(**kwargs)
    checkpoint = load_model(model, model_path, device=device)
    model = model.to(device)
    model.eval()
    return model, checkpoint.get('metrics', {})


def main(args):
    """Main comparison function."""
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    set_seed(config['experiment']['seed'])
    setup_logging(
        log_file='model_comparison.log',
        level=config['logging']['level']
    )
    
    # Create directories
    create_directories([config['paths']['results_dir']])
    
    # Get device
    device = get_device(config['experiment']['device'] == 'cuda')
    
    # Load test dataset
    logging.info(f"Loading test data from {config['data']['train_path']}")
    
    test_dataset = ABADataset(
        data_path=config['data']['train_path'],
        mode='test',
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Dictionary to store models
    models = {}
    model_types = {}
    
    # Load RGCN models if available
    models_dir = Path(config['paths']['models_dir'])
    
    # Try to load RGCN
    rgcn_path = models_dir / 'rgcn_final.pt'
    if rgcn_path.exists():
        logging.info("Loading RGCN model...")
        graph_dataset = ABAGraphDataset(
            data_path=config['data']['train_path'],
            test_size=config['data']['test_size'],
            val_size=config['data']['val_size'],
            random_state=config['data']['random_state']
        )
        
        rgcn_config = config['models']['rgcn']
        rgcn_model, _ = load_trained_model(
            RGCN,
            rgcn_path,
            config,
            device,
            num_nodes=graph_dataset.get_num_nodes(),
            hidden_dim=rgcn_config['hidden_dim'],
            num_layers=rgcn_config['num_layers'],
            num_relations=rgcn_config['num_relations'],
            dropout=rgcn_config['dropout'],
            use_node_features=rgcn_config['use_node_features']
        )
        models['RGCN'] = rgcn_model
        model_types['RGCN'] = 'rgcn'
    
    # Try to load RGCN with Attention
    rgcn_att_path = models_dir / 'rgcn_attention_final.pt'
    if rgcn_att_path.exists():
        logging.info("Loading RGCN with Attention model...")
        if 'graph_dataset' not in locals():
            graph_dataset = ABAGraphDataset(
                data_path=config['data']['train_path'],
                test_size=config['data']['test_size'],
                val_size=config['data']['val_size'],
                random_state=config['data']['random_state']
            )
        
        rgcn_att_config = config['models']['rgcn_attention']
        rgcn_att_model, _ = load_trained_model(
            RGCNWithAttention,
            rgcn_att_path,
            config,
            device,
            num_nodes=graph_dataset.get_num_nodes(),
            hidden_dim=rgcn_att_config['hidden_dim'],
            num_layers=rgcn_att_config['num_layers'],
            num_relations=rgcn_att_config['num_relations'],
            num_heads=rgcn_att_config['num_heads'],
            dropout=rgcn_att_config['dropout'],
            use_node_features=rgcn_att_config['use_node_features']
        )
        models['RGCN-Attention'] = rgcn_att_model
        model_types['RGCN-Attention'] = 'rgcn'
    
    # Try to load BERT
    bert_path = models_dir / 'bert_final.pt'
    if bert_path.exists():
        logging.info("Loading BERT model...")
        bert_config = config['models']['bert']
        bert_model, _ = load_trained_model(
            BERTLinkPredictor,
            bert_path,
            config,
            device,
            model_name=bert_config['model_name'],
            hidden_dim=bert_config['hidden_dim'],
            dropout=bert_config['dropout'],
            freeze_bert=bert_config['freeze_bert'],
            pooling_strategy=bert_config['pooling_strategy']
        )
        models['BERT'] = bert_model
        model_types['BERT'] = 'bert'
    
    # Try to load Cross-Encoder
    cross_encoder_path = models_dir / 'cross_encoder_final.pt'
    if cross_encoder_path.exists():
        logging.info("Loading Cross-Encoder BERT model...")
        cross_config = config['models']['cross_encoder']
        cross_model, _ = load_trained_model(
            CrossEncoderBERT,
            cross_encoder_path,
            config,
            device,
            model_name=cross_config['model_name'],
            dropout=cross_config['dropout'],
            freeze_bert=cross_config['freeze_bert']
        )
        models['Cross-Encoder'] = cross_model
        model_types['Cross-Encoder'] = 'bert'
    
    if not models:
        logging.warning("No trained models found! Please train models first.")
        logging.info("Run the following commands to train models:")
        logging.info("  python train_rgcn.py --model rgcn")
        logging.info("  python train_rgcn.py --model rgcn_attention")
        logging.info("  python train_bert.py --model bert")
        logging.info("  python train_bert.py --model cross_encoder")
        return
    
    # Compare models
    logging.info(f"Comparing {len(models)} models...")
    results = compare_models(models, test_loader, model_types, device)
    
    # Save results
    results_path = Path(config['paths']['results_dir']) / 'model_comparison.json'
    save_results(results, results_path)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1].get('f1', 0))
    logging.info(f"\nBest model (by F1-score): {best_model[0]} with F1={best_model[1]['f1']:.4f}")
    
    logging.info("Model comparison completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare models for ABA link prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='src/aba_link_prediction/config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args)