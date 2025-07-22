import torch
import torch.nn as nn

def train_model(model, data, train_edges, node_to_idx, num_epochs=100, lr=0.001):
    """モデルを学習"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    train_losses = []
    
    # トレーニングデータを準備
    edge_pairs = [(node_to_idx[u], node_to_idx[v]) for (u, v), _ in train_edges]
    labels = torch.tensor([label for _, label in train_edges], dtype=torch.float32)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        predictions = model(data.x, data.edge_index, data.edge_attr, edge_pairs)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return train_losses