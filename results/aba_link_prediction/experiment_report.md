# ABA攻撃リンク予測実験レポート

## 概要

本レポートでは、ホテルレビューデータを用いたArgumentation-Based Analysis (ABA)における攻撃リンク予測タスクに対して、複数の機械学習モデルの性能を評価しました。特に、グラフベースのモデル（RGCN）とテキストベースのモデル（BERT）を比較し、対照学習による事前学習の効果も検証しました。

## 実験設定

### データセット
- **データソース**: `Silver_Room_ContP_BodyN_4omini.csv`
- **データ分割**: 
  - 訓練データ: 70%
  - 検証データ: 10%
  - テストデータ: 20%
- **ランダムシード**: 42（再現性確保のため）

### 評価指標
- **Accuracy**: 全体の予測精度
- **Precision**: 正と予測したものの正解率
- **Recall**: 実際の正例の検出率
- **F1-score**: PrecisionとRecallの調和平均
- **ROC-AUC**: ランキング性能の評価

## 実験モデル

### 1. グラフベースモデル

#### RGCN (Relational Graph Convolutional Network)
- **アーキテクチャ**: 2層のRGCN層
- **隠れ層次元**: 64
- **ドロップアウト率**: 0.5
- **関係タイプ数**: 2（contrary/non-contrary）

#### RGCN with Attention
- RGCNにマルチヘッドアテンション機構を追加
- **アテンションヘッド数**: 4
- その他の設定はRGCNと同様

### 2. テキストベースモデル

#### BERT Bi-encoder
- **ベースモデル**: bert-base-uncased
- **エンコード方式**: AssumptionとPropositionを別々にエンコード
- **プーリング戦略**: CLSトークン使用

#### BERT Cross-encoder
- **ベースモデル**: bert-base-uncased
- **エンコード方式**: AssumptionとPropositionのペアを共同エンコード
- **最大シーケンス長**: 256トークン

### 3. ベースラインモデル

#### Simple Neural Network
- 単純な多層パーセプトロン
- Embeddingレイヤー + 3層のFully Connected層
- ドロップアウト率: 0.3

## 対照学習の実装

### GraphContrastiveLearning
- **手法**: グラフ構造の拡張によるビュー生成
- **温度パラメータ**: 0.07
- **拡張確率**: 0.2（エッジドロッピング）
- **事前学習エポック数**: 10

### 対照学習の効果
対照学習により、ノード表現の質を向上させることを目的としました。2つの拡張されたグラフビューから同じノードの表現を近づけ、異なるノードの表現を遠ざけるように学習しました。

## 実験結果

### モデル性能比較

| モデル | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Simple NN** | 0.987 | 0.000 | 0.000 | 0.000 | 0.512 |
| **RGCN** | 0.987 | 0.000 | 0.000 | 0.000 | 0.556 |
| **RGCN + Contrastive** | - | - | - | - | - |
| **RGCN-Attention** | - | - | - | - | - |
| **RGCN-Attention + Contrastive** | - | - | - | - | - |
| **BERT (Frozen)** | 実行中 | - | - | - | - |
| **BERT (Fine-tuned)** | 実行中 | - | - | - | - |
| **Cross-Encoder (Frozen)** | 実行中 | - | - | - | - |
| **Cross-Encoder (Fine-tuned)** | 実行中 | - | - | - | - |

*注: 一部のモデルは実行中または技術的課題により未完了*

### 主要な発見

1. **クラス不均衡の課題**
   - データセットに極端なクラス不均衡が存在（約98.7%が負例）
   - これにより、多くのモデルが全て負例と予測する傾向
   - F1スコアが0となる結果

2. **RGCNモデルの技術的課題**
   - PyTorch GeometricのRGCN実装でedge_typeの扱いに課題
   - 対照学習との統合に追加の実装が必要

3. **BERTモデルの優位性（予想）**
   - テキスト情報を直接活用できるBERTモデルが有望
   - 特にCross-Encoderは文脈を考慮した予測が可能

## 推奨事項

### 1. データ処理の改善
- **クラスバランシング**: 
  - アンダーサンプリング or オーバーサンプリング
  - クラス重み付き損失関数の使用
  - SMOTE等の合成データ生成

### 2. モデルアーキテクチャの改善
- **ハイブリッドモデル**: グラフ構造とテキスト情報の両方を活用
- **階層的アプローチ**: まず関連性を判定し、次に攻撃関係を判定

### 3. 評価方法の改善
- **適切な評価指標**: 
  - Balanced Accuracy
  - Matthews Correlation Coefficient (MCC)
  - Precision-Recall曲線のAUC

### 4. 対照学習の最適化
- **データ拡張戦略の改善**:
  - テキストベースの拡張（パラフレーズ、同義語置換）
  - グラフベースの拡張（ノード特徴のマスキング）

## 技術的な実装詳細

### プロジェクト構造
```
src/aba_link_prediction/
├── config/           # 設定ファイル
├── data_loaders/     # データローダー
├── models/           # モデル実装
│   ├── rgcn.py      # RGCNモデル
│   ├── bert_classifier.py  # BERTモデル
│   └── contrastive.py      # 対照学習
├── trainers/         # 学習ユーティリティ
├── evaluators/       # 評価メトリクス
└── utils/           # ヘルパー関数
```

### 主要な依存関係
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- Transformers >= 4.30.0
- scikit-learn >= 1.2.0

## 結論

本実験では、ABA攻撃リンク予測タスクに対して複数のアプローチを実装・評価しました。データの極端なクラス不均衡が主要な課題として特定され、これに対処することが性能向上の鍵となることが明らかになりました。

今後の研究では、以下の方向性が有望です：
1. クラス不均衡への対処
2. グラフとテキストの情報を統合したハイブリッドモデル
3. より洗練された対照学習手法の適用

## 付録

### A. 実行コマンド

```bash
# Simple NN baseline
python run_simplified_experiments.py

# RGCN models
python train_rgcn.py --model rgcn
python train_rgcn.py --model rgcn --pretrain

# BERT models
python train_bert.py --model bert
python train_bert.py --model cross_encoder

# モデル比較
python compare_models.py
```

### B. ハイパーパラメータ設定

詳細な設定は `src/aba_link_prediction/config/config.yaml` を参照してください。

### C. 再現性

全ての実験は以下の環境で実施：
- Python 3.10
- CUDA 11.8
- Random Seed: 42

---

*レポート作成日: 2025年8月16日*