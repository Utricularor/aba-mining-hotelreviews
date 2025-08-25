import pickle
import os
import pandas as pd
import networkx as nx
from make_graph_dataset import build_aba_graph

def save_graph_pickle(graph: nx.DiGraph, filename: str):
    """グラフをPickle形式で保存"""
    # outputディレクトリが存在しない場合は作成
    os.makedirs("data/output", exist_ok=True)
    
    filepath = f"data/output/{filename}.pkl"
    
    with open(filepath, 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"✓ グラフを保存しました: {filepath}")
    print(f"  - ノード数: {graph.number_of_nodes()}")
    print(f"  - エッジ数: {graph.number_of_edges()}")
    
    # ファイルサイズも表示
    file_size = os.path.getsize(filepath)
    print(f"  - ファイルサイズ: {file_size:,} bytes")
    
    return filepath

if __name__ == "__main__":
    # データ読み込み
    aba_file_path = "data/input/Original ABA Dataset for Version 2 [June 15] - 1. hotel in Larnaca-Cyprus - Topic.csv"
    cols = ['ReviewID', 'Title', 'Topic', 'Pos/Neg', 'Claim', 'Head',
    'Body 1', 'Body 2', 'Body 3', 'Body 4', 'Body 5', 'Body 6', 'Body 7',
    'Body 8', 'Body 9', 'Body 10', 'Body 11', 'Body 12', 'Body 13',
    'Body 14', 'Body 15', 'Cont. Body 1', 'Cont. Body 2', 'Cont. Body 3',
    'Cont. Body 4', 'Cont. Body 5', 'Cont. Body 6', 'Cont. Body 7', 
    'Cont. Body 8', 'Cont. Body 9', 'Cont. Body 10', 'Cont. Body 11',
    'Cont. Body 12', 'Cont. Body 13', 'Cont. Body 14', 'Cont. Body 15',]
    aba = pd.read_csv(aba_file_path, usecols=cols)
    aba_room = aba[aba['Topic'] == 'Room']
    
    # 2つのcontraファイルを読み込み
    contra_file_path_1 = "data/output/Silver_Room_ContP_BodyN_4omini.csv"
    contra_file_path_2 = "data/output/Silver_Room_ContN_BodyP_4omini.csv"
    
    print(f"📂 Contraファイル1を読み込み: {contra_file_path_1}")
    contra_1 = pd.read_csv(contra_file_path_1)
    print(f"  - データ数: {len(contra_1)} 行")
    
    print(f"📂 Contraファイル2を読み込み: {contra_file_path_2}")
    contra_2 = pd.read_csv(contra_file_path_2)
    print(f"  - データ数: {len(contra_2)} 行")
    
    # 2つのcontraデータを統合
    contra_combined = pd.concat([contra_1, contra_2], ignore_index=True)
    print(f"📊 統合後のデータ数: {len(contra_combined)} 行")
    
    # 重複チェック（もしあれば）
    duplicates = contra_combined.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  重複データ: {duplicates} 行")
        contra_combined = contra_combined.drop_duplicates().reset_index(drop=True)
        print(f"✓ 重複除去後: {len(contra_combined)} 行")
    else:
        print("✓ 重複データなし")

    # グラフ構築
    aba_graph_room = build_aba_graph(aba_room, contra_combined)

    # グラフ保存（統合データを反映したファイル名）
    saved_file = save_graph_pickle(aba_graph_room, "aba_graph_staff_combined")
    print(f"\n保存完了: {saved_file}")
    print(f"📈 統合されたcontraデータを使用したグラフが保存されました")
