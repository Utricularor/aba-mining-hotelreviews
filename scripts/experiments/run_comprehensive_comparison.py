#!/usr/bin/env python3
"""Comprehensive comparison of all ABA link prediction approaches."""

import os
import sys
import json
import logging
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.aba_link_prediction.utils import set_seed, create_directories

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_existing_results() -> Dict:
    """Load all existing experiment results."""
    results = {}
    
    result_files = {
        'baseline': '../../results/aba_link_prediction/experiment_results.json',
        'balanced': '../../results/aba_link_prediction/balanced_experiment_results.json',
        'hard_negative': '../../results/aba_link_prediction/hard_negative_results.json',
        'robust': '../../results/aba_link_prediction/robust_experiment_results.json'
    }
    
    for exp_name, filepath in result_files.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                results[exp_name] = data
                logger.info(f"Loaded {exp_name} results from {filepath}")
        else:
            logger.warning(f"Could not find {filepath}")
    
    return results

def create_comparison_table(results: Dict) -> pd.DataFrame:
    """Create a comprehensive comparison table of all experiments."""
    
    rows = []
    
    # Define mappings for cleaner names
    experiment_names = {
        'baseline': 'Baseline (Imbalanced)',
        'balanced': 'Balanced (Undersampling)',
        'hard_negative': 'Hard Negative Sampling',
        'robust': 'Robust (Balanced)'
    }
    
    model_names = {
        'RGCN': 'RGCN',
        'RGCN_Attention': 'RGCN w/ Attention',
        'RGCN_Contrastive': 'RGCN + Contrastive',
        'RGCN_Attention_Contrastive': 'RGCN Att. + Contrastive',
        'BERT': 'BERT Bi-Encoder',
        'CrossEncoder': 'BERT Cross-Encoder',
        'BERT_Balanced': 'BERT Bi-Encoder',
        'RGCN_Balanced': 'RGCN',
        'SimpleNN_Balanced': 'Simple NN',
        'SimpleNN_HardNeg': 'Simple NN',
        'BERT_HardNeg': 'BERT Bi-Encoder',
        'CrossEncoder_HardNeg': 'BERT Cross-Encoder',
        'SimpleNN_Balanced_Robust': 'Simple NN',
        'BERT_Balanced_Robust': 'BERT Bi-Encoder',
        'CrossEncoder_Balanced_Robust': 'BERT Cross-Encoder'
    }
    
    for exp_name, exp_data in results.items():
        exp_display_name = experiment_names.get(exp_name, exp_name)
        
        for model_key, model_data in exp_data.items():
            model_display_name = model_names.get(model_key, model_key)
            
            metrics = model_data.get('metrics', {})
            
            row = {
                'Experiment': exp_display_name,
                'Model': model_display_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1 Score': metrics.get('f1', 0),
                'ROC-AUC': metrics.get('roc_auc', 0)
            }
            
            # Add confusion matrix stats if available
            cm = model_data.get('confusion_matrix', [])
            if cm and len(cm) >= 2:
                tn = cm[0][0] if len(cm[0]) > 0 else 0
                fp = cm[0][1] if len(cm[0]) > 1 else 0
                fn = cm[1][0] if len(cm[1]) > 0 else 0
                tp = cm[1][1] if len(cm[1]) > 1 else 0
                
                row['True Positives'] = tp
                row['True Negatives'] = tn
                row['False Positives'] = fp
                row['False Negatives'] = fn
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by F1 score descending
    df = df.sort_values('F1 Score', ascending=False)
    
    return df

def create_comprehensive_visualization(results: Dict, df: pd.DataFrame):
    """Create comprehensive visualization comparing all approaches."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. F1 Score comparison across all experiments
    ax1 = plt.subplot(3, 3, 1)
    top_models = df.nlargest(10, 'F1 Score')
    bars = ax1.barh(range(len(top_models)), top_models['F1 Score'].values)
    ax1.set_yticks(range(len(top_models)))
    ax1.set_yticklabels([f"{row['Model']} ({row['Experiment'][:10]}...)" 
                         for _, row in top_models.iterrows()], fontsize=8)
    ax1.set_xlabel('F1 Score')
    ax1.set_title('Top 10 Models by F1 Score', fontweight='bold')
    
    # Color bars by experiment type
    colors = {'Baseline': 'red', 'Balanced': 'blue', 'Hard Negative': 'green', 'Robust': 'purple'}
    for i, (_, row) in enumerate(top_models.iterrows()):
        exp_type = row['Experiment'].split()[0]
        color = colors.get(exp_type, 'gray')
        bars[i].set_color(color)
    
    # 2. ROC-AUC comparison
    ax2 = plt.subplot(3, 3, 2)
    top_models_auc = df.nlargest(10, 'ROC-AUC')
    bars = ax2.barh(range(len(top_models_auc)), top_models_auc['ROC-AUC'].values)
    ax2.set_yticks(range(len(top_models_auc)))
    ax2.set_yticklabels([f"{row['Model']} ({row['Experiment'][:10]}...)" 
                         for _, row in top_models_auc.iterrows()], fontsize=8)
    ax2.set_xlabel('ROC-AUC')
    ax2.set_title('Top 10 Models by ROC-AUC', fontweight='bold')
    
    # 3. Precision-Recall scatter plot
    ax3 = plt.subplot(3, 3, 3)
    for exp_name in df['Experiment'].unique():
        exp_df = df[df['Experiment'] == exp_name]
        ax3.scatter(exp_df['Recall'], exp_df['Precision'], 
                   label=exp_name, s=100, alpha=0.7)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall Trade-off', fontweight='bold')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 Score by Experiment Type
    ax4 = plt.subplot(3, 3, 4)
    exp_f1_means = df.groupby('Experiment')['F1 Score'].mean().sort_values(ascending=False)
    ax4.bar(range(len(exp_f1_means)), exp_f1_means.values)
    ax4.set_xticks(range(len(exp_f1_means)))
    ax4.set_xticklabels(exp_f1_means.index, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Mean F1 Score')
    ax4.set_title('Average F1 Score by Experiment Type', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Model Performance Heatmap
    ax5 = plt.subplot(3, 3, 5)
    pivot_df = df.pivot_table(values='F1 Score', 
                              index='Model', 
                              columns='Experiment', 
                              aggfunc='first')
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                ax=ax5, cbar_kws={'label': 'F1 Score'})
    ax5.set_title('F1 Score Heatmap (Model × Experiment)', fontweight='bold')
    ax5.set_xlabel('')
    ax5.set_ylabel('')
    
    # 6. ROC-AUC by Model Type
    ax6 = plt.subplot(3, 3, 6)
    model_auc_means = df.groupby('Model')['ROC-AUC'].mean().sort_values(ascending=False)
    ax6.bar(range(len(model_auc_means)), model_auc_means.values)
    ax6.set_xticks(range(len(model_auc_means)))
    ax6.set_xticklabels(model_auc_means.index, rotation=45, ha='right', fontsize=9)
    ax6.set_ylabel('Mean ROC-AUC')
    ax6.set_title('Average ROC-AUC by Model Type', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Best F1 vs Accuracy scatter
    ax7 = plt.subplot(3, 3, 7)
    # Group by model type and plot
    model_types = df['Model'].unique()
    colors_model = plt.cm.tab10(np.linspace(0, 1, len(model_types)))
    for i, model in enumerate(model_types):
        model_df = df[df['Model'] == model]
        ax7.scatter(model_df['Accuracy'], model_df['F1 Score'], 
                   label=model, s=100, alpha=0.7, color=colors_model[i])
    ax7.set_xlabel('Accuracy')
    ax7.set_ylabel('F1 Score')
    ax7.set_title('Accuracy vs F1 Score', fontweight='bold')
    ax7.legend(fontsize=7, loc='best', ncol=2)
    ax7.grid(True, alpha=0.3)
    
    # 8. Confusion Matrix Statistics
    ax8 = plt.subplot(3, 3, 8)
    if 'True Positives' in df.columns:
        # Calculate TPR and TNR for models with confusion matrix data
        df_cm = df[df['True Positives'].notna()].copy()
        df_cm['TPR'] = df_cm['True Positives'] / (df_cm['True Positives'] + df_cm['False Negatives'])
        df_cm['TNR'] = df_cm['True Negatives'] / (df_cm['True Negatives'] + df_cm['False Positives'])
        
        top_balanced = df_cm.nlargest(10, 'F1 Score')
        x = np.arange(len(top_balanced))
        width = 0.35
        
        bars1 = ax8.bar(x - width/2, top_balanced['TPR'].values, width, label='TPR (Sensitivity)')
        bars2 = ax8.bar(x + width/2, top_balanced['TNR'].values, width, label='TNR (Specificity)')
        
        ax8.set_xticks(x)
        ax8.set_xticklabels([f"{row['Model'][:8]}..." 
                             for _, row in top_balanced.iterrows()], 
                            rotation=45, ha='right', fontsize=8)
        ax8.set_ylabel('Rate')
        ax8.set_title('TPR vs TNR for Top Models', fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    # Create summary statistics
    summary_data = []
    summary_data.append(['Best F1 Score', f"{df['F1 Score'].max():.4f}"])
    best_f1_model = df.loc[df['F1 Score'].idxmax()]
    summary_data.append(['Best F1 Model', f"{best_f1_model['Model']} ({best_f1_model['Experiment']})"])
    summary_data.append(['Best ROC-AUC', f"{df['ROC-AUC'].max():.4f}"])
    best_auc_model = df.loc[df['ROC-AUC'].idxmax()]
    summary_data.append(['Best AUC Model', f"{best_auc_model['Model']} ({best_auc_model['Experiment']})"])
    summary_data.append(['Total Experiments', str(len(df))])
    summary_data.append(['Unique Models', str(df['Model'].nunique())])
    summary_data.append(['Avg F1 (Balanced)', f"{df[df['Experiment'].str.contains('Balanced')]['F1 Score'].mean():.4f}"])
    summary_data.append(['Avg F1 (Baseline)', f"{df[df['Experiment'].str.contains('Baseline')]['F1 Score'].mean():.4f}"])
    
    table = ax9.table(cellText=summary_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax9.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('Comprehensive ABA Link Prediction Results Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('../../results/aba_link_prediction/comprehensive_comparison.png', 
                dpi=150, bbox_inches='tight')
    logger.info("Saved comprehensive visualization")
    
    return fig

def generate_markdown_report(df: pd.DataFrame):
    """Generate a detailed markdown report of all experiments."""
    
    report = []
    report.append("# ABA Link Prediction - Comprehensive Experiment Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    best_f1 = df.loc[df['F1 Score'].idxmax()]
    best_auc = df.loc[df['ROC-AUC'].idxmax()]
    
    report.append(f"- **Best F1 Score**: {best_f1['F1 Score']:.4f} "
                 f"({best_f1['Model']} - {best_f1['Experiment']})")
    report.append(f"- **Best ROC-AUC**: {best_auc['ROC-AUC']:.4f} "
                 f"({best_auc['Model']} - {best_auc['Experiment']})")
    report.append(f"- **Total Experiments Run**: {len(df)}")
    report.append(f"- **Number of Unique Models**: {df['Model'].nunique()}\n")
    
    # Key Findings
    report.append("## Key Findings\n")
    
    # Compare baseline vs balanced
    baseline_mean_f1 = df[df['Experiment'].str.contains('Baseline')]['F1 Score'].mean()
    balanced_mean_f1 = df[df['Experiment'].str.contains('Balanced')]['F1 Score'].mean()
    
    if balanced_mean_f1 > 0 and baseline_mean_f1 > 0:
        improvement = ((balanced_mean_f1 - baseline_mean_f1) / baseline_mean_f1) * 100
        report.append(f"1. **Class Balancing Impact**: Balancing improved average F1 score "
                     f"by {improvement:.1f}% (from {baseline_mean_f1:.4f} to {balanced_mean_f1:.4f})")
    
    # Best performing model type
    model_f1_means = df.groupby('Model')['F1 Score'].mean()
    best_model_type = model_f1_means.idxmax()
    report.append(f"2. **Best Model Architecture**: {best_model_type} "
                 f"achieved the highest average F1 score ({model_f1_means.max():.4f})")
    
    # Effect of hard negatives
    if 'Hard Negative' in df['Experiment'].values:
        hard_neg_mean = df[df['Experiment'].str.contains('Hard Negative')]['F1 Score'].mean()
        report.append(f"3. **Hard Negative Sampling**: Average F1 score of {hard_neg_mean:.4f}")
    
    report.append("")
    
    # Detailed Results by Experiment
    report.append("## Detailed Results by Experiment\n")
    
    for exp_name in df['Experiment'].unique():
        report.append(f"### {exp_name}\n")
        
        exp_df = df[df['Experiment'] == exp_name].sort_values('F1 Score', ascending=False)
        
        # Create markdown table
        report.append("| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |")
        report.append("|-------|----------|-----------|--------|----------|---------|")
        
        for _, row in exp_df.iterrows():
            report.append(f"| {row['Model']} | "
                        f"{row['Accuracy']:.4f} | "
                        f"{row['Precision']:.4f} | "
                        f"{row['Recall']:.4f} | "
                        f"{row['F1 Score']:.4f} | "
                        f"{row['ROC-AUC']:.4f} |")
        
        report.append("")
    
    # Model Comparison
    report.append("## Model Architecture Comparison\n")
    
    model_stats = df.groupby('Model').agg({
        'F1 Score': ['mean', 'std', 'max'],
        'ROC-AUC': ['mean', 'std', 'max'],
        'Accuracy': 'mean'
    }).round(4)
    
    report.append("| Model | Mean F1 | Std F1 | Max F1 | Mean AUC | Std AUC | Max AUC | Mean Acc |")
    report.append("|-------|---------|--------|--------|----------|---------|---------|----------|")
    
    for model in model_stats.index:
        report.append(f"| {model} | "
                    f"{model_stats.loc[model, ('F1 Score', 'mean')]:.4f} | "
                    f"{model_stats.loc[model, ('F1 Score', 'std')]:.4f} | "
                    f"{model_stats.loc[model, ('F1 Score', 'max')]:.4f} | "
                    f"{model_stats.loc[model, ('ROC-AUC', 'mean')]:.4f} | "
                    f"{model_stats.loc[model, ('ROC-AUC', 'std')]:.4f} | "
                    f"{model_stats.loc[model, ('ROC-AUC', 'max')]:.4f} | "
                    f"{model_stats.loc[model, ('Accuracy', 'mean')]:.4f} |")
    
    report.append("")
    
    # Recommendations
    report.append("## Recommendations\n")
    
    report.append("Based on the comprehensive experiments, we recommend:\n")
    report.append(f"1. **For Production Use**: {best_f1['Model']} with {best_f1['Experiment']} "
                 f"approach (F1: {best_f1['F1 Score']:.4f})")
    report.append(f"2. **For High Recall Requirements**: Consider models with balanced datasets")
    report.append(f"3. **For Computational Efficiency**: Simple NN models provide competitive "
                 f"performance with faster training times")
    
    # Technical Details
    report.append("\n## Technical Details\n")
    report.append("- **Dataset**: Silver_Room_ContP_BodyN_4omini.csv")
    report.append("- **Class Distribution**: Originally 1.31% positive, 98.69% negative")
    report.append("- **Balancing Strategies**: Undersampling, Hard Negative Sampling")
    report.append("- **Evaluation**: 5-fold cross-validation, held-out test set")
    report.append("- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC")
    
    # Save report
    report_path = '../../results/aba_link_prediction/comprehensive_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Saved comprehensive report to {report_path}")
    
    return '\n'.join(report)

def generate_latex_table(df: pd.DataFrame):
    """Generate LaTeX table for paper inclusion."""
    
    # Select top models
    top_models = df.nlargest(10, 'F1 Score')
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Top 10 ABA Link Prediction Models}")
    latex.append("\\label{tab:aba_results}")
    latex.append("\\begin{tabular}{llccccc}")
    latex.append("\\hline")
    latex.append("Experiment & Model & Acc. & Prec. & Rec. & F1 & AUC \\\\")
    latex.append("\\hline")
    
    for _, row in top_models.iterrows():
        exp_short = row['Experiment'].replace('(', '').replace(')', '').split()[0]
        latex.append(f"{exp_short} & {row['Model']} & "
                    f"{row['Accuracy']:.3f} & "
                    f"{row['Precision']:.3f} & "
                    f"{row['Recall']:.3f} & "
                    f"{row['F1 Score']:.3f} & "
                    f"{row['ROC-AUC']:.3f} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    latex_path = '../../results/aba_link_prediction/results_table.tex'
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    
    logger.info(f"Saved LaTeX table to {latex_path}")
    
    return '\n'.join(latex)

def main():
    """Run comprehensive comparison analysis."""
    
    set_seed(42)
    create_directories(['../../results/aba_link_prediction'])
    
    logger.info("="*60)
    logger.info("ABA LINK PREDICTION - COMPREHENSIVE COMPARISON")
    logger.info("="*60)
    
    # Load all existing results
    all_results = load_existing_results()
    
    if not all_results:
        logger.error("No experiment results found. Please run experiments first.")
        return
    
    # Create comparison table
    df = create_comparison_table(all_results)
    
    # Save as CSV
    csv_path = '../../results/aba_link_prediction/comprehensive_comparison.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison table to {csv_path}")
    
    # Create visualizations
    create_comprehensive_visualization(all_results, df)
    
    # Generate reports
    generate_markdown_report(df)
    generate_latex_table(df)
    
    # Print summary to console
    logger.info("\n" + "="*60)
    logger.info("TOP 5 MODELS BY F1 SCORE")
    logger.info("="*60)
    
    top5 = df.nlargest(5, 'F1 Score')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        logger.info(f"\n{i}. {row['Model']} ({row['Experiment']})")
        logger.info(f"   F1 Score: {row['F1 Score']:.4f}")
        logger.info(f"   ROC-AUC: {row['ROC-AUC']:.4f}")
        logger.info(f"   Precision: {row['Precision']:.4f}")
        logger.info(f"   Recall: {row['Recall']:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT TYPE COMPARISON")
    logger.info("="*60)
    
    exp_comparison = df.groupby('Experiment')[['F1 Score', 'ROC-AUC']].mean().round(4)
    exp_comparison = exp_comparison.sort_values('F1 Score', ascending=False)
    
    for exp_name in exp_comparison.index:
        logger.info(f"\n{exp_name}:")
        logger.info(f"  Average F1 Score: {exp_comparison.loc[exp_name, 'F1 Score']:.4f}")
        logger.info(f"  Average ROC-AUC: {exp_comparison.loc[exp_name, 'ROC-AUC']:.4f}")
    
    logger.info("\n✅ Comprehensive comparison completed successfully!")
    logger.info(f"📊 Results saved to: results/aba_link_prediction/")
    logger.info(f"📈 Visualization: comprehensive_comparison.png")
    logger.info(f"📝 Report: comprehensive_report.md")
    logger.info(f"📊 Data: comprehensive_comparison.csv")
    
    return df

if __name__ == '__main__':
    df = main()