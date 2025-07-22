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
    aba_file_path = "data/Original ABA Dataset for Version 2 [June 15] - 1. hotel in Larnaca-Cyprus - Topic.csv"
    cols = ['ReviewID', 'Title', 'Topic', 'Pos/Neg', 'Claim', 'Head',
    'Body 1', 'Body 2', 'Body 3', 'Body 4', 'Body 5', 'Body 6', 'Body 7',
    'Body 8', 'Body 9', 'Body 10', 'Body 11', 'Body 12', 'Body 13',
    'Body 14', 'Body 15', 'Cont. Body 1', 'Cont. Body 2', 'Cont. Body 3',
    'Cont. Body 4', 'Cont. Body 5', 'Cont. Body 6', 'Cont. Body 7', 
    'Cont. Body 8', 'Cont. Body 9', 'Cont. Body 10', 'Cont. Body 11',
    'Cont. Body 12', 'Cont. Body 13', 'Cont. Body 14', 'Cont. Body 15',]
    aba = pd.read_csv(aba_file_path, usecols=cols)
    aba_room = aba[aba['Topic'] == 'Room']
    
    contra_file_path = "data/Room_Contrary(P)Body(N)_4omini.csv"
    contra = pd.read_csv(contra_file_path)

    # グラフ構築
    aba_graph_room = build_aba_graph(aba_room, contra)

    # グラフ保存
    saved_file = save_graph_pickle(aba_graph_room, "aba_graph_room")
    print(f"\n保存完了: {saved_file}")
