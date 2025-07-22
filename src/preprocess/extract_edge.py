import pickle
import networkx as nx

def create_inference_only_graph(original_graph: nx.DiGraph):
    """inferenceエッジのみを使用したグラフを作成"""
    inference_graph = nx.DiGraph()
    
    # 全てのノードを追加（属性も含む）
    for node, attr in original_graph.nodes(data=True):
        inference_graph.add_node(node, **attr)
    
    # inferenceエッジのみを追加
    inference_edges = []
    for u, v, d in original_graph.edges(data=True):
        if d.get('relation') == 'inference':
            inference_graph.add_edge(u, v, relation='inference')
            inference_edges.append((u, v))
    print(f"Inference グラフ: ノード数={inference_graph.number_of_nodes()}, エッジ数={inference_graph.number_of_edges()}")
    return inference_graph, inference_edges

def collect_attack_edges(original_graph: nx.DiGraph):
    """攻撃エッジを収集（予測対象）"""
    attack_edges = []
    for u, v, d in original_graph.edges(data=True):
        if d.get('relation') == 'attack':
            attack_edges.append((u, v))
    
    print(f"Attack エッジ数: {len(attack_edges)}")
    return attack_edges

if __name__ == "__main__":
    # データの読み込み
    filepath = 'data/output/aba_graph_room.pkl'
    with open(filepath, 'rb') as f:
        original_graph = pickle.load(f)

    # グラフの作成
    inference_graph, inference_edges = create_inference_only_graph(original_graph)
    attack_edges = collect_attack_edges(original_graph)

    # グラフの保存
    with open('data/output/inference_graph_room.pkl', 'wb') as f:
        pickle.dump(inference_graph, f)
    with open('data/output/attack_edges_room.pkl', 'wb') as f:
        pickle.dump(attack_edges, f)