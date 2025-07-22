#!/bin/bash

# ABA Mining 開発環境セットアップスクリプト

echo "🚀 ABA Mining 開発環境をセットアップ中..."

# 仮想環境が有効になっているかチェック
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 仮想環境が有効: $VIRTUAL_ENV"
else
    echo "⚠️  仮想環境を有効化してください: source .venv/bin/activate"
    exit 1
fi

# 開発用依存関係をインストール
echo "📦 開発用パッケージをインストール中..."
pip install -e ".[dev]"

# IPython プロファイルディレクトリを作成
echo "🔧 IPython プロファイル設定中..."
IPYTHON_DIR="$HOME/.ipython/profile_default/startup"
mkdir -p "$IPYTHON_DIR"

# startup script をコピー
if [ -f ".ipython/profile_default/startup/00_autoreload.py" ]; then
    cp .ipython/profile_default/startup/00_autoreload.py "$IPYTHON_DIR/"
    echo "✅ 自動リロード設定をコピーしました"
else
    echo "⚠️  startup script が見つかりません"
fi

# Jupyter拡張を有効化
echo "🎯 Jupyter拡張を設定中..."
jupyter contrib nbextension install --user --skip-running-check 2>/dev/null || echo "   (contrib nbextensions がない場合はスキップ)"

echo ""
echo "🎉 開発環境のセットアップが完了しました！"
echo ""
echo "📝 使用方法:"
echo "   1. Jupyter Notebook/Lab を起動:"
echo "      jupyter notebook --config=jupyter_config.py"
echo "      または"
echo "      jupyter lab --config=jupyter_config.py"
echo ""
echo "   2. ノートブック内で自動リロードが有効になります"
echo "   3. ソースコードを変更すると自動的にモジュールがリロードされます"
echo ""
echo "🔄 手動でリロードしたい場合:"
echo "   %load_ext autoreload"
echo "   %autoreload 2" 