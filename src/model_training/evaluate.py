import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, data, test_edges, node_to_idx, device: str = 'cpu'):
    """モデルを評価"""
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        edge_pairs = [(node_to_idx[u], node_to_idx[v]) for (u, v), _ in test_edges]
        x_dev = data.x.to(device)
        edge_index_dev = data.edge_index.to(device)
        edge_type_dev = data.edge_attr.to(device) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        predictions = model(x_dev, edge_index_dev, edge_type_dev, edge_pairs)
        predictions_np = predictions.detach().cpu().numpy()
    
    y_true = [label for _, label in test_edges]
    y_pred = (predictions_np > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, predictions_np) if len(set(y_true)) > 1 else 0
    }
    
    return metrics, predictions_np

def evaluate_baseline(baseline, test_edges):
    """ベースラインを評価"""
    edge_pairs = [(u, v) for (u, v), _ in test_edges]
    predictions = baseline.predict(edge_pairs)
    
    y_true = [label for _, label in test_edges]
    y_pred = (predictions > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, predictions) if len(set(y_true)) > 1 else 0
    }
    
    return metrics, predictions