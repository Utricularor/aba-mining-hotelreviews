"""
開発用ヘルパースクリプト

Jupyter Notebook内で自動リロード機能を簡単に有効化するためのスクリプトです。
ノートブックの最初のセルで以下を実行してください：

```python
%run notebook/dev_setup.py
```
"""

def setup_autoreload():
    """自動リロード機能を有効化"""
    try:
        from IPython import get_ipython
        
        # autoreload 拡張を読み込む
        get_ipython().run_line_magic('load_ext', 'autoreload')
        
        # autoreload モード2を設定（すべてのモジュールを自動リロード）
        get_ipython().run_line_magic('autoreload', '2')
        
        print("🔄 自動リロード機能が有効になりました")
        print("   📝 ソースコードを変更すると自動的にモジュールがリロードされます")
        print("   🔧 手動でリロードしたい場合: %reload_ext autoreload")
        
        return True
        
    except Exception as e:
        print(f"⚠️  自動リロード設定中にエラーが発生しました: {e}")
        print("   手動で以下を実行してください:")
        print("   %load_ext autoreload")
        print("   %autoreload 2")
        return False

def setup_matplotlib():
    """matplotlib の日本語設定"""
    try:
        import matplotlib.pyplot as plt
        import japanize_matplotlib
        
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 12
        
        print("📊 matplotlib の日本語表示が有効になりました")
        return True
        
    except Exception as e:
        print(f"⚠️  matplotlib設定中にエラーが発生しました: {e}")
        return False

def setup_development_environment():
    """開発環境を一括でセットアップ"""
    print("🚀 ABA Mining 開発環境をセットアップ中...")
    print()
    
    # 自動リロード設定
    autoreload_ok = setup_autoreload()
    print()
    
    # matplotlib設定
    matplotlib_ok = setup_matplotlib()
    print()
    
    if autoreload_ok and matplotlib_ok:
        print("✅ 開発環境のセットアップが完了しました！")
        print()
        print("📋 使用可能な機能:")
        print("   • 自動リロード: ソースコード変更時の自動モジュール更新")
        print("   • 日本語matplotlib: グラフの日本語表示")
        print()
        print("💡 ヒント:")
        print("   • from src.xxx import xxx でモジュールをインポート")
        print("   • ソースコードを変更後、セルを再実行するだけでOK")
        print("   • カーネル再起動は不要です")
    else:
        print("⚠️  一部の設定でエラーが発生しました")

# スクリプトが直接実行された場合（%run で呼び出された場合）
if __name__ == "__main__":
    setup_development_environment() 