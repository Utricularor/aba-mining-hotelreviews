"""
実験結果の可視化と統計分析
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォント対応
from scipy.stats import ttest_rel
import json
import os
from typing import Dict, List, Any


def convert_to_serializable(obj):
    """
    numpy型をJSON serializable な型に変換
    
    Args:
        obj: 変換する対象オブジェクト
    
    Returns:
        JSON serializable なオブジェクト
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def calculate_statistics(results: Dict[str, Dict[str, List[float]]]) -> Dict:
    """
    結果の統計量を計算
    
    Args:
        results: モデルごとの評価結果
    
    Returns:
        stats: 統計量
    """
    stats = {}
    
    for model_name, metrics in results.items():
        stats[model_name] = {}
        for metric_name, values in metrics.items():
            if len(values) > 0:
                stats[model_name][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'ci_lower': float(np.percentile(values, 2.5)),
                    'ci_upper': float(np.percentile(values, 97.5))
                }
            else:
                stats[model_name][metric_name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'ci_lower': 0.0,
                    'ci_upper': 0.0
                }
    
    return stats


def perform_statistical_tests(results: Dict[str, Dict[str, List[float]]]) -> Dict:
    """
    統計的有意性検定を実行（対応のあるt検定）
    
    Args:
        results: モデルごとの評価結果
    
    Returns:
        test_results: 検定結果
    """
    test_results = {}
    models = list(results.keys())
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            test_results[f"{model1}_vs_{model2}"] = {}
            
            for metric in ['accuracy', 'f1', 'auc']:
                values1 = results[model1][metric]
                values2 = results[model2][metric]
                
                # データ長が同じかチェック
                min_len = min(len(values1), len(values2))
                if min_len < 2:
                    test_results[f"{model1}_vs_{model2}"][metric] = {
                        'statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False
                    }
                    continue
                
                # 配列を同じ長さに調整
                values1_trimmed = np.array(values1[:min_len])
                values2_trimmed = np.array(values2[:min_len])
                
                try:
                    # 対応のあるt検定
                    statistic, p_value = ttest_rel(values1_trimmed, values2_trimmed)
                    test_results[f"{model1}_vs_{model2}"][metric] = {
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    test_results[f"{model1}_vs_{model2}"][metric] = {
                        'statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False
                    }
    
    return test_results


def plot_training_curves(train_losses: List[float], val_losses: List[float] = None,
                         model_name: str = "Model", save_path: str = None,
                         show_plot: bool = True):
    """
    学習曲線をプロット
    
    Args:
        train_losses: 訓練損失のリスト
        val_losses: 検証損失のリスト
        model_name: モデル名
        save_path: 保存先パス
        show_plot: プロットを表示するか
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None and len(val_losses) > 0:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title(f'{model_name} - 学習曲線', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最終損失値を表示
    if train_losses:
        final_train_loss = train_losses[-1]
        plt.text(0.02, 0.98, f'最終訓練損失: {final_train_loss:.4f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    if val_losses and len(val_losses) > 0:
        final_val_loss = val_losses[-1]
        plt.text(0.02, 0.88, f'最終検証損失: {final_val_loss:.4f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 学習曲線を保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_box_plots(results: Dict[str, Dict[str, List[float]]],
                   save_path: str = None, show_plot: bool = True):
    """
    ボックスプロットで結果を可視化
    
    Args:
        results: モデルごとの評価結果
        save_path: 保存先パス
        show_plot: プロットを表示するか
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('5-fold Cross-Validation 結果', fontsize=16)
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        # データを準備
        model_names = list(results.keys())
        values = [results[model][metric] for model in model_names]
        
        # ボックスプロット
        box_plot = ax.boxplot(values, labels=model_names, patch_artist=True)
        
        # 色を設定
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{label} の分布')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ボックスプロットを保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_bar_charts(stats: Dict, save_path: str = None, show_plot: bool = True):
    """
    棒グラフでAUCスコアを比較
    
    Args:
        stats: 統計量
        save_path: 保存先パス
        show_plot: プロットを表示するか
    """
    plt.figure(figsize=(12, 6))
    
    model_names = list(stats.keys())
    auc_means = [stats[model]['auc']['mean'] for model in model_names]
    auc_stds = [stats[model]['auc']['std'] for model in model_names]
    
    # 色設定
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(model_names, auc_means, yerr=auc_stds, capsize=5, 
                   alpha=0.7, color=colors)
    
    # 最高性能モデルを強調
    best_idx = np.argmax(auc_means)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    plt.title('AUC スコア比較 (平均 ± 標準偏差)', fontsize=14)
    plt.ylabel('AUC')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 棒グラフを保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_comprehensive_analysis(results: Dict[str, Dict[str, List[float]]],
                                save_path: str = None, show_plot: bool = True):
    """
    包括的な分析プロット
    
    Args:
        results: モデルごとの評価結果
        save_path: 保存先パス
        show_plot: プロットを表示するか
    """
    plt.figure(figsize=(15, 10))
    
    models = list(results.keys())
    
    # サブプロット1: パフォーマンス比較
    plt.subplot(2, 3, 1)
    accuracies = [np.mean(results[model]['accuracy']) for model in models]
    accuracy_stds = [np.std(results[model]['accuracy']) for model in models]
    
    bars = plt.bar(models, accuracies, yerr=accuracy_stds, capsize=5, alpha=0.7)
    plt.title('モデル精度比較')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 最高性能モデルを強調
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2)
    
    # サブプロット2: F1スコア比較
    plt.subplot(2, 3, 2)
    f1_scores = [np.mean(results[model]['f1']) for model in models]
    f1_stds = [np.std(results[model]['f1']) for model in models]
    
    plt.bar(models, f1_scores, yerr=f1_stds, capsize=5, alpha=0.7, color='lightgreen')
    plt.title('F1スコア比較')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # サブプロット3: AUC比較
    plt.subplot(2, 3, 3)
    auc_scores = [np.mean(results[model]['auc']) for model in models]
    auc_stds = [np.std(results[model]['auc']) for model in models]
    
    plt.bar(models, auc_scores, yerr=auc_stds, capsize=5, alpha=0.7, color='lightcoral')
    plt.title('AUCスコア比較')
    plt.ylabel('AUC')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # サブプロット4: 精度分布のヒストグラム
    plt.subplot(2, 3, 4)
    for i, model in enumerate(models):
        plt.hist(results[model]['accuracy'], alpha=0.6, label=model, bins=5)
    plt.title('精度分布')
    plt.xlabel('Accuracy')
    plt.ylabel('頻度')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # サブプロット5: 各フォールドでの性能変動
    plt.subplot(2, 3, 5)
    n_folds = len(results[models[0]]['accuracy'])
    folds = range(1, n_folds + 1)
    for model in models:
        plt.plot(folds, results[model]['accuracy'], 'o-', label=model, alpha=0.7)
    plt.title('フォールド間の精度変動')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # サブプロット6: 精度 vs F1 の散布図
    plt.subplot(2, 3, 6)
    for model in models:
        acc_values = results[model]['accuracy']
        f1_values = results[model]['f1']
        plt.scatter(acc_values, f1_values, label=model, alpha=0.7, s=50)
    plt.title('精度 vs F1スコア')
    plt.xlabel('Accuracy')
    plt.ylabel('F1 Score')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('学習結果の包括的分析', fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 包括的分析プロットを保存: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def display_results_table(stats: Dict, test_results: Dict = None):
    """
    結果を表形式で表示
    
    Args:
        stats: 統計量
        test_results: 統計的検定結果
    """
    print("\n📊 Cross-Validation 結果 (平均 ± 標準偏差)")
    print("=" * 90)
    print(f"{'モデル':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12}")
    print("-" * 90)
    
    for model_name, model_stats in stats.items():
        row = f"{model_name:<25}"
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            mean = model_stats[metric]['mean']
            std = model_stats[metric]['std']
            row += f"{mean:.3f}±{std:.3f}  "
        print(row)
    
    if test_results:
        print("\n📈 統計的有意性検定結果 (p-value)")
        print("=" * 70)
        for comparison, metrics in test_results.items():
            model1, model2 = comparison.split('_vs_')
            print(f"\n{model1} vs {model2}:")
            for metric, result in metrics.items():
                p_value = result['p_value']
                if np.isnan(p_value):
                    print(f"  {metric}: p=N/A (データ不足)")
                else:
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"  {metric}: p={p_value:.4f} {significance}")


def save_results_to_file(results: Dict, stats: Dict, test_results: Dict,
                         output_dir: str, config: Dict):
    """
    結果をファイルに保存
    
    Args:
        results: モデルごとの評価結果
        stats: 統計量
        test_results: 統計的検定結果
        output_dir: 出力ディレクトリ
        config: 設定辞書
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 結果をJSON形式で保存（numpy型を変換）
    results_summary = {
        'experiment_type': 'robust_cross_validation',
        'config': config,
        'raw_results': results,
        'statistics': stats,
        'statistical_tests': test_results
    }
    
    # numpy型をPython型に変換
    results_summary = convert_to_serializable(results_summary)
    
    output_file = os.path.join(output_dir, 'experiment_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 実験結果を保存: {output_file}")

