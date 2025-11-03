"""
Named Models Experiment Runner

以下の6モデルを固定セットとして5-fold CVで学習・評価し、結果を保存・可視化します。

- FreezedBertRgcnMlp
- FreezedBertMlp
- FinetunedBertMlp
- FinetunedBertCosSim
- TfidfLr
- Random

備考:
- 対照学習（contrastive learning）は使用しません（再現性確保）。
- 設定は既存の robust_experiment.yaml を流用します（ハイパラ・入出力位置など）。
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

# プロジェクトルートをパスに追加（スクリプト直実行対応）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.cross_validation import create_cross_validation_splits
from src.experiments.run_robust_experiment import (
    load_config,
    determine_experiment_id,
    setup_output_directory,
    prepare_data,
    generate_negatives,
)
from src.model_defs.models import (
    FreezedBertRgcnMlp,
    FreezedBertMlp,
    FinetunedBertMlp,
    FinetunedBertCosSim,
    TfidfLr,
    Random,
)
from src.model_training.train import train_model
from src.model_training.train_bert import (
    ABADataset,
    train_bert_model,
    evaluate_bert_model,
)
from src.model_training.evaluate import evaluate_model, evaluate_baseline
from src.visualization.plot_results import (
    calculate_statistics,
    perform_statistical_tests,
    plot_box_plots,
    plot_bar_charts,
    plot_comprehensive_analysis,
    display_results_table,
    save_results_to_file,
)


def run_named_models_experiment(config_path: str, args=None):
    # 設定の読み込み
    config = load_config(config_path)

    # 実験ID（デフォルト名を補助）
    experiment_id = determine_experiment_id(config, args)
    if experiment_id is None or experiment_id.startswith("exp_"):
        experiment_id = experiment_id or ""
    # 名前の明示
    if not experiment_id:
        experiment_id = "named_models"
    else:
        experiment_id = f"{experiment_id}_named_models"

    # 出力先
    output_dir = setup_output_directory(config, experiment_id)
    config['data']['output_dir'] = output_dir
    config['data']['experiment_id'] = experiment_id

    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"\n💻 使用デバイス: {device}")

    # データ準備（BERTノード埋め込みを生成し特徴に使用）
    (
        original_graph,
        inference_graph,
        attack_edges,
        all_nodes,
        node_embeddings,
        node_to_idx,
        embedding_matrix,
        data,
    ) = prepare_data(config)

    # ネガティブサンプリング
    all_negatives = generate_negatives(
        original_graph,
        all_nodes,
        attack_edges,
        embedding_matrix,
        node_to_idx,
        inference_graph,
        config,
    )

    # CV分割
    cv_splits = create_cross_validation_splits(
        attack_edges,
        all_negatives,
        n_splits=int(config['cross_validation']['n_folds']),
        seed=int(config['data']['seed']),
    )

    # 結果入れ物
    model_names = [
        'FreezedBertRgcnMlp',
        'FreezedBertMlp',
        'FinetunedBertMlp',
        'FinetunedBertCosSim',
        'TfidfLr',
        'Random',
    ]
    results: Dict[str, Dict[str, List[float]]] = {
        name: {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']} for name in model_names
    }

    print(f"\n{'='*70}")
    print("固定セット（6モデル）で5-fold Cross-Validation を開始...")
    print(f"対象モデル: {', '.join(model_names)}")
    print(f"{'='*70}\n")

    # BERT設定（ImprovedBERT設定を流用）
    bert_cfg = config['models']['improved_bert']
    val_split_ratio = float(config['cross_validation']['val_split_ratio'])

    for fold_idx, (train_edges, test_edges) in enumerate(cv_splits):
        print(f"\n📊 Fold {fold_idx + 1}/{len(cv_splits)}")
        print("-" * 50)

        # 1) FreezedBertRgcnMlp（R-GCN）
        print("🔥 FreezedBertRgcnMlp を学習中...")
        rgcn_cfg = config['models']['rgcn']
        rgcn_model = FreezedBertRgcnMlp(
            input_dim=data.x.shape[1],
            hidden_dim=int(rgcn_cfg['hidden_dim']),
            num_layers=int(rgcn_cfg['num_layers']),
            num_relations=1,
        )
        # train/val split
        train_size = int((1 - val_split_ratio) * len(train_edges))
        shuffled = train_edges.copy()
        import random as _random
        _random.shuffle(shuffled)
        rgcn_train_edges = shuffled[:train_size]
        rgcn_val_edges = shuffled[train_size:]

        _ = train_model(
            rgcn_model,
            data,
            rgcn_train_edges,
            node_to_idx,
            num_epochs=int(rgcn_cfg['num_epochs']),
            lr=float(rgcn_cfg['learning_rate']),
            model_name="FreezedBertRgcnMlp",
            verbose=bool(rgcn_cfg.get('verbose', True)),
            validation_edges=rgcn_val_edges,
        )
        metrics, _preds = evaluate_model(rgcn_model, data, test_edges, node_to_idx)
        for k, v in metrics.items():
            results['FreezedBertRgcnMlp'][k].append(v)
        print(f"結果: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")

        # 2) FreezedBertMlp
        print("\n🤖 FreezedBertMlp を学習中...")
        ds_train = ABADataset(train_edges, all_nodes)
        ds_test = ABADataset(test_edges, all_nodes)
        tr_size = int((1 - val_split_ratio) * len(ds_train))
        va_size = len(ds_train) - tr_size
        if tr_size >= 1 and va_size >= 1:
            tr_subset, va_subset = torch.utils.data.random_split(ds_train, [tr_size, va_size])
        else:
            tr_subset, va_subset = ds_train, ds_test
        dl_tr = DataLoader(tr_subset, batch_size=int(bert_cfg['batch_size']), shuffle=True)
        dl_va = DataLoader(va_subset, batch_size=int(bert_cfg['val_batch_size']), shuffle=False)
        dl_te = DataLoader(ds_test, batch_size=int(bert_cfg['val_batch_size']), shuffle=False)
        model_freezed = FreezedBertMlp(
            model_name=bert_cfg['model_name'],
            dropout=float(bert_cfg['dropout']),
            max_length=int(bert_cfg['max_length']),
            device=str(device),
        )
        sched = None
        if bert_cfg.get('scheduler'):
            sched = {
                'type': bert_cfg['scheduler']['type'],
                'step_size': int(bert_cfg['scheduler']['step_size']),
                'gamma': float(bert_cfg['scheduler']['gamma']),
            }
        _ = train_bert_model(
            model_freezed,
            dl_tr,
            dl_va,
            num_epochs=int(bert_cfg['num_epochs']),
            lr=float(bert_cfg['learning_rate']),
            device=str(device),
            model_name=f"FreezedBertMlp (Fold {fold_idx+1})",
            early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
            verbose=True,
            scheduler_config=sched,
        )
        met, _ = evaluate_bert_model(model_freezed, dl_te, device=str(device))
        for k, v in met.items():
            results['FreezedBertMlp'][k].append(v)
        print(f"結果: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

        # 3) FinetunedBertMlp
        print("\n🤖 FinetunedBertMlp を学習中...")
        model_finetuned_mlp = FinetunedBertMlp(
            model_name=bert_cfg['model_name'],
            dropout=float(bert_cfg['dropout']),
            max_length=int(bert_cfg['max_length']),
            device=str(device),
        )
        _ = train_bert_model(
            model_finetuned_mlp,
            dl_tr,
            dl_va,
            num_epochs=int(bert_cfg['num_epochs']),
            lr=float(bert_cfg['learning_rate']),
            device=str(device),
            model_name=f"FinetunedBertMlp (Fold {fold_idx+1})",
            early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
            verbose=True,
            scheduler_config=sched,
        )
        met, _ = evaluate_bert_model(model_finetuned_mlp, dl_te, device=str(device))
        for k, v in met.items():
            results['FinetunedBertMlp'][k].append(v)
        print(f"結果: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

        # 4) FinetunedBertCosSim（シアミーズ + コサインロジット）
        print("\n🤖 FinetunedBertCosSim を学習中...")
        model_cos = FinetunedBertCosSim(
            model_name=bert_cfg['model_name'],
            max_length=int(bert_cfg['max_length']),
            device=str(device),
        )
        _ = train_bert_model(
            model_cos,
            dl_tr,
            dl_va,
            num_epochs=int(bert_cfg['num_epochs']),
            lr=float(bert_cfg['learning_rate']),
            device=str(device),
            model_name=f"FinetunedBertCosSim (Fold {fold_idx+1})",
            early_stopping_patience=int(bert_cfg.get('early_stopping_patience', 5)),
            verbose=True,
            scheduler_config=sched,
        )
        met, _ = evaluate_bert_model(model_cos, dl_te, device=str(device))
        for k, v in met.items():
            results['FinetunedBertCosSim'][k].append(v)
        print(f"結果: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

        # 5) TfidfLr
        print("\n📝 TfidfLr を学習・評価中...")
        tfidf = TfidfLr()
        tfidf.fit(train_edges, all_nodes)
        met, _ = evaluate_baseline(tfidf, test_edges)
        for k, v in met.items():
            results['TfidfLr'][k].append(v)
        print(f"結果: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

        # 6) Random
        print("\n🎲 Random を評価中...")
        rnd = Random()
        met, _ = evaluate_baseline(rnd, test_edges)
        for k, v in met.items():
            results['Random'][k].append(v)
        print(f"結果: Acc={met['accuracy']:.3f}, F1={met['f1']:.3f}, AUC={met['auc']:.3f}")

    # 統計・表示・保存・可視化
    stats = calculate_statistics(results)
    tests = perform_statistical_tests(results)
    display_results_table(stats, tests)

    # 可視化
    vis = config.get('visualization', {"enabled": True, "plots": ["box_plots", "bar_charts", "comprehensive_analysis"], "show_plots": False})
    if vis.get('enabled', True):
        if 'box_plots' in vis.get('plots', []):
            plot_box_plots(results, save_path=os.path.join(output_dir, 'box_plots.png'), show_plot=vis.get('show_plots', False))
        if 'bar_charts' in vis.get('plots', []):
            plot_bar_charts(stats, save_path=os.path.join(output_dir, 'bar_charts.png'), show_plot=vis.get('show_plots', False))
        if 'comprehensive_analysis' in vis.get('plots', []):
            plot_comprehensive_analysis(results, save_path=os.path.join(output_dir, 'comprehensive_analysis.png'), show_plot=vis.get('show_plots', False))

    save_results_to_file(results, stats, tests, output_dir, config)
    print(f"\n✅ 完了: 出力は {output_dir} に保存しました")


def main():
    parser = argparse.ArgumentParser(
        description='Run fixed set of named models (6 models) with 5-fold CV.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/robust_experiment.yaml',
        help='Path to configuration file (default: config/robust_experiment.yaml)'
    )
    parser.add_argument(
        '--experiment-id',
        type=str,
        default=None,
        dest='experiment_id',
        help='Base experiment ID (will be suffixed with _named_models)'
    )
    args = parser.parse_args()
    run_named_models_experiment(args.config, args)


if __name__ == '__main__':
    main()


