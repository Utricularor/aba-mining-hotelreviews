"""
Cross-Validation 実行モジュール
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Any

from ..model_defs.models import (
    AttackLinkPredictor,
    RandomBaseline,
    BERTCosineSimilarityBaseline,
    TFIDFLogisticRegressionBaseline,
    ImprovedBERTLinkPredictor,
    CrossEncoderBERTLinkPredictor
)
from ..model_training.train import train_model
from ..model_training.train_bert import (
    train_bert_model, 
    evaluate_bert_model, 
    ABADataset
)
from ..model_training.evaluate import evaluate_model, evaluate_baseline


def create_cross_validation_splits(
    attack_edges: List[Tuple],
    negative_edges: List[Tuple],
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[List, List]]:
    """
    エッジレベルランダム分割によるクロスバリデーション用の分割を作成
    
    Args:
        attack_edges: Attackエッジのリスト
        negative_edges: ネガティブサンプルのリスト
        n_splits: 分割数
        seed: ランダムシード
    
    Returns:
        List of (train_edges, test_edges) tuples
    """
    random.seed(seed)
    
    # 全エッジにラベルを付与
    all_edges_with_labels = []
    for edge in attack_edges:
        all_edges_with_labels.append((edge, 1))  # Positive
    for edge in negative_edges:
        all_edges_with_labels.append((edge, 0))  # Negative
    
    # エッジをランダムシャッフル
    random.shuffle(all_edges_with_labels)
    
    # k-fold分割を作成
    fold_size = len(all_edges_with_labels) // n_splits
    cv_splits = []
    
    for test_fold_idx in range(n_splits):
        start_idx = test_fold_idx * fold_size
        if test_fold_idx == n_splits - 1:
            # 最後のフォールドは残り全て
            end_idx = len(all_edges_with_labels)
        else:
            end_idx = (test_fold_idx + 1) * fold_size
        
        test_edges = all_edges_with_labels[start_idx:end_idx]
        train_edges = all_edges_with_labels[:start_idx] + all_edges_with_labels[end_idx:]
        
        cv_splits.append((train_edges, test_edges))
    
    return cv_splits


def run_cross_validation(
    cv_splits: List[Tuple[List, List]],
    data: Any,  # PyTorch Geometric Data
    node_to_idx: Dict,
    all_nodes: List[str],
    node_embeddings: Dict,
    config: Dict,
    device: str = 'cpu'
) -> Dict[str, Dict[str, List[float]]]:
    """
    Cross-validationを実行
    
    Args:
        cv_splits: Cross-validation分割
        data: グラフデータ
        node_to_idx: ノードインデックスマッピング
        all_nodes: 全ノードのリスト
        node_embeddings: ノード埋め込み
        config: 設定辞書
        device: 計算デバイス
    
    Returns:
        results: モデルごとの評価結果
    """
    # 結果を保存する辞書
    results = {}
    
    # 有効なモデルを特定
    enabled_models = []
    if config['models']['rgcn']['enabled']:
        enabled_models.append('AttackLinkPredictor')
    if config['models']['improved_bert']['enabled']:
        enabled_models.append('ImprovedBERT')
    if config['models']['cross_encoder_bert']['enabled']:
        enabled_models.append('CrossEncoderBERT')
    if config['models']['random_baseline']['enabled']:
        enabled_models.append('Random')
    if config['models']['bert_cosine']['enabled']:
        enabled_models.append('BERTCosine')
    if config['models']['tfidf_lr']['enabled']:
        enabled_models.append('TFIDF+LR')
    
    # 結果辞書の初期化
    for model_name in enabled_models:
        results[model_name] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
    
    print(f"\n{'='*70}")
    print(f"5-fold Cross-Validation を開始...")
    print(f"有効なモデル: {', '.join(enabled_models)}")
    print(f"{'='*70}\n")
    
    for fold_idx, (train_edges, test_edges) in enumerate(cv_splits):
        print(f"\n📊 Fold {fold_idx + 1}/{len(cv_splits)}")
        print("-" * 50)
        
        # 1. Attack Link Predictor (R-GCN)
        if 'AttackLinkPredictor' in enabled_models:
            print("🔥 Attack Link Predictor (R-GCN) を学習中...")
            
            rgcn_config = config['models']['rgcn']
            embedding_dim = data.x.shape[1]
            
            model = AttackLinkPredictor(
                input_dim=embedding_dim,
                hidden_dim=rgcn_config['hidden_dim'],
                num_layers=rgcn_config['num_layers']
            )
            
            # 訓練データから検証データを分離
            val_split_ratio = config['cross_validation']['val_split_ratio']
            train_size = int((1 - val_split_ratio) * len(train_edges))
            
            shuffled_train_edges = train_edges.copy()
            random.shuffle(shuffled_train_edges)
            
            rgcn_train_edges = shuffled_train_edges[:train_size]
            rgcn_val_edges = shuffled_train_edges[train_size:]
            
            # 学習
            training_info = train_model(
                model, data, rgcn_train_edges, node_to_idx,
                num_epochs=rgcn_config['num_epochs'],
                lr=rgcn_config['learning_rate'],
                model_name="Attack Link Predictor (R-GCN)",
                verbose=rgcn_config.get('verbose', True),
                validation_edges=rgcn_val_edges
            )
            
            # 評価
            metrics, _ = evaluate_model(model, data, test_edges, node_to_idx)
            for metric_name, value in metrics.items():
                results['AttackLinkPredictor'][metric_name].append(value)
            
            print(f"結果: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
        
        # 2. Improved BERT
        if 'ImprovedBERT' in enabled_models:
            print("\n🤖 Improved BERT (固定BERT + 線形層) を学習中...")
            
            try:
                bert_config = config['models']['improved_bert']
                
                # データセット準備
                train_dataset = ABADataset(train_edges, all_nodes)
                test_dataset = ABADataset(test_edges, all_nodes)
                
                # 訓練データから検証データを分離
                val_split_ratio = config['cross_validation']['val_split_ratio']
                train_size = int((1 - val_split_ratio) * len(train_dataset))
                val_size = len(train_dataset) - train_size
                
                if train_size >= 1 and val_size >= 1:
                    train_subset, val_subset = torch.utils.data.random_split(
                        train_dataset, [train_size, val_size]
                    )
                else:
                    train_subset = train_dataset
                    val_subset = test_dataset
                
                # データローダー（型を明示的に変換）
                train_loader = DataLoader(
                    train_subset,
                    batch_size=int(bert_config['batch_size']),
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_subset,
                    batch_size=int(bert_config['val_batch_size']),
                    shuffle=False
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=int(bert_config['val_batch_size']),
                    shuffle=False
                )
                
                # モデル初期化（型を明示的に変換）
                bert_model = ImprovedBERTLinkPredictor(
                    model_name=bert_config['model_name'],
                    dropout=float(bert_config['dropout']),
                    max_length=int(bert_config['max_length']),
                    freeze_bert=bool(bert_config['freeze_bert']),
                    device=device
                )
                
                # 学習（型を明示的に変換）
                scheduler_config = None
                if bert_config.get('scheduler'):
                    scheduler_config = {
                        'type': bert_config['scheduler']['type'],
                        'step_size': int(bert_config['scheduler']['step_size']),
                        'gamma': float(bert_config['scheduler']['gamma'])
                    }
                
                training_info = train_bert_model(
                    bert_model,
                    train_loader,
                    val_loader,
                    num_epochs=int(bert_config['num_epochs']),
                    lr=float(bert_config['learning_rate']),
                    device=device,
                    model_name=f"Improved BERT (Fold {fold_idx+1})",
                    early_stopping_patience=int(bert_config.get('early_stopping_patience', 5)),
                    verbose=True,
                    scheduler_config=scheduler_config
                )
                
                # 評価
                bert_metrics, _ = evaluate_bert_model(bert_model, test_loader, device)
                for metric_name, value in bert_metrics.items():
                    results['ImprovedBERT'][metric_name].append(value)
                
                print(f"結果: Acc={bert_metrics['accuracy']:.3f}, "
                      f"F1={bert_metrics['f1']:.3f}, AUC={bert_metrics['auc']:.3f}")
                
                # メモリクリーンアップ
                del bert_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ Improved BERT 学習でエラー: {e}")
                # ダミー値を追加
                for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                    results['ImprovedBERT'][metric_name].append(0.0)
        
        # 3. Cross-Encoder BERT
        if 'CrossEncoderBERT' in enabled_models:
            print("\n🤖 Cross-Encoder BERT を学習中...")
            
            try:
                bert_config = config['models']['cross_encoder_bert']
                
                # データセット準備
                train_dataset = ABADataset(train_edges, all_nodes)
                test_dataset = ABADataset(test_edges, all_nodes)
                
                # 訓練データから検証データを分離
                val_split_ratio = config['cross_validation']['val_split_ratio']
                train_size = int((1 - val_split_ratio) * len(train_dataset))
                val_size = len(train_dataset) - train_size
                
                if train_size >= 1 and val_size >= 1:
                    train_subset, val_subset = torch.utils.data.random_split(
                        train_dataset, [train_size, val_size]
                    )
                else:
                    train_subset = train_dataset
                    val_subset = test_dataset
                
                # データローダー（型を明示的に変換）
                train_loader = DataLoader(
                    train_subset,
                    batch_size=int(bert_config['batch_size']),
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_subset,
                    batch_size=int(bert_config['val_batch_size']),
                    shuffle=False
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=int(bert_config['val_batch_size']),
                    shuffle=False
                )
                
                # モデル初期化（型を明示的に変換）
                bert_model = CrossEncoderBERTLinkPredictor(
                    model_name=bert_config['model_name'],
                    dropout=float(bert_config['dropout']),
                    max_length=int(bert_config['max_length']),
                    freeze_bert=bool(bert_config['freeze_bert']),
                    device=device
                )
                
                # 学習（型を明示的に変換）
                scheduler_config = None
                if bert_config.get('scheduler'):
                    scheduler_config = {
                        'type': bert_config['scheduler']['type'],
                        'step_size': int(bert_config['scheduler']['step_size']),
                        'gamma': float(bert_config['scheduler']['gamma'])
                    }
                
                training_info = train_bert_model(
                    bert_model,
                    train_loader,
                    val_loader,
                    num_epochs=int(bert_config['num_epochs']),
                    lr=float(bert_config['learning_rate']),
                    device=device,
                    model_name=f"Cross-Encoder BERT (Fold {fold_idx+1})",
                    early_stopping_patience=int(bert_config.get('early_stopping_patience', 5)),
                    verbose=True,
                    scheduler_config=scheduler_config
                )
                
                # 評価
                bert_metrics, _ = evaluate_bert_model(bert_model, test_loader, device)
                for metric_name, value in bert_metrics.items():
                    results['CrossEncoderBERT'][metric_name].append(value)
                
                print(f"結果: Acc={bert_metrics['accuracy']:.3f}, "
                      f"F1={bert_metrics['f1']:.3f}, AUC={bert_metrics['auc']:.3f}")
                
                # メモリクリーンアップ
                del bert_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ Cross-Encoder BERT 学習でエラー: {e}")
                # ダミー値を追加
                for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                    results['CrossEncoderBERT'][metric_name].append(0.0)
        
        # 4. Random Baseline
        if 'Random' in enabled_models:
            print("\n🎲 Random Baseline を評価中...")
            random_baseline = RandomBaseline()
            metrics, _ = evaluate_baseline(random_baseline, test_edges)
            for metric_name, value in metrics.items():
                results['Random'][metric_name].append(value)
            
            print(f"結果: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
        
        # 5. BERT Cosine Similarity Baseline
        if 'BERTCosine' in enabled_models:
            print("\n🤖 BERT Cosine Similarity Baseline を評価中...")
            bert_baseline = BERTCosineSimilarityBaseline(node_embeddings, node_to_idx)
            metrics, _ = evaluate_baseline(bert_baseline, test_edges)
            for metric_name, value in metrics.items():
                results['BERTCosine'][metric_name].append(value)
            
            print(f"結果: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
        
        # 6. TF-IDF + Logistic Regression Baseline
        if 'TFIDF+LR' in enabled_models:
            print("\n📝 TF-IDF + Logistic Regression Baseline を学習・評価中...")
            tfidf_baseline = TFIDFLogisticRegressionBaseline()
            tfidf_baseline.fit(train_edges, all_nodes)
            metrics, _ = evaluate_baseline(tfidf_baseline, test_edges)
            for metric_name, value in metrics.items():
                results['TFIDF+LR'][metric_name].append(value)
            
            print(f"結果: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
    
    print(f"\n{'='*70}")
    print("✅ 5-fold Cross-Validation 完了!")
    print(f"{'='*70}\n")
    
    return results

