import torch
import torch.nn as nn
import time

def train_model(model, data, train_edges, node_to_idx, num_epochs=100, lr=0.001, 
                model_name="R-GCN", verbose=True, validation_edges=None, device: str = 'cpu'):
    """
    拡張されたモデル学習関数（詳細な学習過程記録付き）
    
    Args:
        model: 学習するモデル
        data: グラフデータ
        train_edges: 訓練エッジ
        node_to_idx: ノードインデックスマッピング
        num_epochs: エポック数
        lr: 学習率
        model_name: モデル名（表示用）
        verbose: 詳細表示フラグ
        validation_edges: 検証エッジ（オプション）
    
    Returns:
        dict: 学習過程の情報（損失、時間など）
    """
    if verbose:
        # モデルパラメータ情報表示
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 {model_name} パラメータ情報:")
        print(f"  総パラメータ数: {total_params:,}")
        print(f"  学習可能パラメータ数: {trainable_params:,}")
        
        # 学習設定表示
        print(f"\n⚙️  {model_name} 学習設定:")
        print(f"  エポック数: {num_epochs}")
        print(f"  学習率: {lr}")
        print(f"  最適化手法: Adam")
        print(f"  損失関数: Binary Cross Entropy")
        print(f"  訓練サンプル数: {len(train_edges)}")
        if validation_edges:
            print(f"  検証サンプル数: {len(validation_edges)}")
        print("-" * 50)
    
    # デバイス設定
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    train_losses = []
    val_losses = []
    
    # トレーニングデータを準備
    edge_pairs = [(node_to_idx[u], node_to_idx[v]) for (u, v), _ in train_edges]
    labels = torch.tensor([label for _, label in train_edges], dtype=torch.float32, device=device)
    # グラフデータをデバイスへ
    x_dev = data.x.to(device)
    edge_index_dev = data.edge_index.to(device)
    edge_type_dev = data.edge_attr.to(device) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    
    # 検証データを準備（もしあれば）
    val_edge_pairs = None
    val_labels = None
    if validation_edges:
        val_edge_pairs = [(node_to_idx[u], node_to_idx[v]) for (u, v), _ in validation_edges]
        val_labels = torch.tensor([label for _, label in validation_edges], dtype=torch.float32, device=device)
    
    if verbose:
        print(f"\n🚀 {model_name} 学習開始...")
        print("=" * 60)
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training step
        optimizer.zero_grad()
        
        predictions = model(x_dev, edge_index_dev, edge_type_dev, edge_pairs)
        train_loss = criterion(predictions, labels)
        
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        
        # Validation step (if validation data is provided)
        val_loss = None
        if validation_edges:
            model.eval()
            with torch.no_grad():
                val_predictions = model(x_dev, edge_index_dev, edge_type_dev, val_edge_pairs)
                val_loss = criterion(val_predictions, val_labels).item()
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            model.train()
        
        epoch_time = time.time() - epoch_start_time
        
        # 詳細な進捗表示
        if verbose and (epoch % 20 == 0 or epoch == num_epochs - 1):
            if val_loss is not None:
                print(f"  📊 Epoch {epoch:3d}/{num_epochs}: "
                      f"Train Loss = {train_loss.item():.4f}, "
                      f"Val Loss = {val_loss:.4f}, "
                      f"Time = {epoch_time:.2f}s")
            else:
                print(f"  📊 Epoch {epoch:3d}/{num_epochs}: "
                      f"Train Loss = {train_loss.item():.4f}, "
                      f"Time = {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    if verbose:
        print("-" * 60)
        print(f"✅ {model_name} 学習完了!")
        print(f"   総学習時間: {total_time:.2f}秒")
        print(f"   最終訓練損失: {train_losses[-1]:.4f}")
        if val_losses:
            print(f"   最終検証損失: {val_losses[-1]:.4f}")
            print(f"   最良検証損失: {best_val_loss:.4f}")
        print(f"   平均エポック時間: {total_time/num_epochs:.3f}秒")
    
    # 学習結果をまとめて返す
    training_info = {
        'train_losses': train_losses,
        'val_losses': val_losses if val_losses else None,
        'total_time': total_time,
        'best_val_loss': best_val_loss if val_losses else None,
        'final_train_loss': train_losses[-1],
        'model_name': model_name,
        'num_epochs': num_epochs,
        'learning_rate': lr
    }
    
    return training_info