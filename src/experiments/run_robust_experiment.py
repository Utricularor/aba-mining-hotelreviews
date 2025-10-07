"""
Robust Experiment 実行スクリプト
"""

import os
import sys
import pickle
import random
import argparse
import yaml
import numpy as np
import torch
from torch_geometric.data import Data

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.preprocess.extract_edge import create_inference_only_graph, collect_attack_edges
from src.preprocess.embed_node import generate_bert_embeddings
from src.augmentation.generate_negative import (
    generate_hard_negatives,
    generate_structural_negatives,
    generate_random_negatives
)
from src.experiments.cross_validation import create_cross_validation_splits, run_cross_validation
from src.visualization.plot_results import (
    calculate_statistics,
    perform_statistical_tests,
    plot_box_plots,
    plot_bar_charts,
    plot_comprehensive_analysis,
    display_results_table,
    save_results_to_file
)


def set_seed(seed: int = 42):
    """
    ランダムシードを設定
    
    Args:
        seed: シード値
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """
    設定ファイルを読み込む
    
    Args:
        config_path: 設定ファイルのパス
    
    Returns:
        config: 設定辞書
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """
    データの準備
    
    Args:
        config: 設定辞書
    
    Returns:
        tuple: (original_graph, inference_graph, attack_edges, all_nodes, 
                node_embeddings, node_to_idx, embedding_matrix, data)
    """
    print("\n" + "="*70)
    print("データの準備を開始...")
    print("="*70)
    
    # データの読み込み
    filepath = config['data']['input_graph']
    print(f"\n📂 グラフデータを読み込み: {filepath}")
    
    with open(filepath, 'rb') as f:
        original_graph = pickle.load(f)
    
    # グラフの作成
    print("\n🔗 グラフ構造の抽出...")
    inference_graph, inference_edges = create_inference_only_graph(original_graph)
    attack_edges = collect_attack_edges(original_graph)
    
    print(f"  Inference グラフ: ノード数={inference_graph.number_of_nodes()}, "
          f"エッジ数={len(inference_edges)}")
    print(f"  Attack エッジ数: {len(attack_edges)}")
    
    # 全ノードを取得
    all_nodes = sorted({n for n in original_graph.nodes()})
    print(f"  総ノード数: {len(all_nodes)}")
    
    # BERTエンベディング生成
    print("\n🤖 BERTエンベディング生成中...")
    node_embeddings = generate_bert_embeddings(all_nodes)
    
    embedding_dim = len(list(node_embeddings.values())[0])
    print(f"  エンベディング次元: {embedding_dim}")
    
    # エンベディング行列を作成
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    embedding_matrix = np.array([node_embeddings[node] for node in all_nodes])
    print(f"  エンベディング行列形状: {embedding_matrix.shape}")
    
    # PyTorch Geometric用データ作成
    print("\n🔧 PyTorch Geometricデータを作成...")
    x = torch.tensor(embedding_matrix, dtype=torch.float32)
    
    edge_index = []
    edge_type = []
    
    for u, v in inference_graph.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        edge_index.append([u_idx, v_idx])
        edge_type.append(0)  # inference = 0
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type)
    print(f"  グラフデータ: ノード数={data.x.size(0)}, エッジ数={data.edge_index.size(1)}")
    
    return (original_graph, inference_graph, attack_edges, all_nodes,
            node_embeddings, node_to_idx, embedding_matrix, data)


def generate_negatives(original_graph, all_nodes, attack_edges, 
                      embedding_matrix, node_to_idx, inference_graph, config):
    """
    ネガティブサンプリング
    
    Args:
        original_graph: 元のグラフ
        all_nodes: 全ノード
        attack_edges: Attackエッジ
        embedding_matrix: エンベディング行列
        node_to_idx: ノードインデックスマッピング
        inference_graph: Inferenceグラフ
        config: 設定辞書
    
    Returns:
        all_negatives: 全ネガティブサンプル
    """
    print("\n" + "="*70)
    print("ネガティブサンプリングを実行...")
    print("="*70)
    
    neg_config = config['negative_sampling']
    
    # Hard negatives
    hard_negatives = []
    if neg_config['hard_negatives']['enabled']:
        print("\n🎯 Hard negatives を生成中...")
        hard_negatives = generate_hard_negatives(
            original_graph, all_nodes, attack_edges,
            embedding_matrix, node_to_idx
        )
        print(f"  生成数: {len(hard_negatives)}")
    
    # Structural negatives
    structural_negatives = []
    if neg_config['structural_negatives']['enabled']:
        print("\n🏗️  Structural negatives を生成中...")
        structural_negatives = generate_structural_negatives(
            original_graph, attack_edges, inference_graph
        )
        print(f"  生成数: {len(structural_negatives)}")
    
    # Random negatives
    random_negatives = []
    if neg_config['random_negatives']['enabled']:
        print("\n🎲 Random negatives を生成中...")
        random_negatives = generate_random_negatives(
            original_graph, attack_edges, all_nodes
        )
        print(f"  生成数: {len(random_negatives)}")
    
    # 全てのネガティブサンプルを結合
    all_negatives = hard_negatives + structural_negatives + random_negatives
    
    print(f"\n📊 ネガティブサンプリング結果:")
    print(f"  Hard negatives: {len(hard_negatives)}")
    print(f"  Structural negatives: {len(structural_negatives)}")
    print(f"  Random negatives: {len(random_negatives)}")
    print(f"  Total negatives: {len(all_negatives)}")
    print(f"  Positive samples: {len(attack_edges)}")
    print(f"  Negative/Positive ratio: {len(all_negatives)/len(attack_edges):.2f}")
    
    return all_negatives


def run_robust_experiment(config_path: str):
    """
    Robust experimentを実行
    
    Args:
        config_path: 設定ファイルのパス
    """
    # 設定の読み込み
    config = load_config(config_path)
    
    # シード設定
    set_seed(config['data']['seed'])
    
    # デバイス設定
    if config['compute']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['compute']['device'])
    
    print(f"\n💻 使用デバイス: {device}")
    
    # データの準備
    (original_graph, inference_graph, attack_edges, all_nodes,
     node_embeddings, node_to_idx, embedding_matrix, data) = prepare_data(config)
    
    # ネガティブサンプリング
    all_negatives = generate_negatives(
        original_graph, all_nodes, attack_edges,
        embedding_matrix, node_to_idx, inference_graph, config
    )
    
    # Cross-validation分割
    print("\n" + "="*70)
    print("Cross-validation分割を作成...")
    print("="*70)
    
    cv_splits = create_cross_validation_splits(
        attack_edges,
        all_negatives,
        n_splits=config['cross_validation']['n_folds'],
        seed=config['data']['seed']
    )
    
    print(f"\n📊 Cross-validation分割結果:")
    for i, (train_edges, test_edges) in enumerate(cv_splits):
        train_pos = sum(1 for _, label in train_edges if label == 1)
        train_neg = sum(1 for _, label in train_edges if label == 0)
        test_pos = sum(1 for _, label in test_edges if label == 1)
        test_neg = sum(1 for _, label in test_edges if label == 0)
        print(f"  Fold {i+1}: Train({train_pos}+, {train_neg}-), Test({test_pos}+, {test_neg}-)")
    
    # Cross-validation実行
    results = run_cross_validation(
        cv_splits,
        data,
        node_to_idx,
        all_nodes,
        node_embeddings,
        config,
        device=str(device)
    )
    
    # 統計分析
    print("\n" + "="*70)
    print("統計分析を実行...")
    print("="*70)
    
    stats = calculate_statistics(results)
    test_results = perform_statistical_tests(results)
    
    # 結果の表示
    display_results_table(stats, test_results)
    
    # 可視化
    if config['visualization']['enabled']:
        print("\n" + "="*70)
        print("結果を可視化...")
        print("="*70)
        
        output_dir = config['data']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        show_plots = config['visualization']['show_plots']
        
        if 'box_plots' in config['visualization']['plots']:
            plot_box_plots(
                results,
                save_path=os.path.join(output_dir, 'box_plots.png'),
                show_plot=show_plots
            )
        
        if 'bar_charts' in config['visualization']['plots']:
            plot_bar_charts(
                stats,
                save_path=os.path.join(output_dir, 'bar_charts.png'),
                show_plot=show_plots
            )
        
        if 'comprehensive_analysis' in config['visualization']['plots']:
            plot_comprehensive_analysis(
                results,
                save_path=os.path.join(output_dir, 'comprehensive_analysis.png'),
                show_plot=show_plots
            )
    
    # 結果の保存
    save_results_to_file(results, stats, test_results, 
                        config['data']['output_dir'], config)
    
    print("\n" + "="*70)
    print("✅ 実験が正常に完了しました！")
    print("="*70)


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description='Robust Experiment for Attack Link Prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='config/robust_experiment.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # 実験実行
    run_robust_experiment(args.config)


if __name__ == '__main__':
    main()

