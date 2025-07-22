import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class AttackLinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_relations=1):
        super().__init__()
        self.num_layers = num_layers
        
        # R-GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(input_dim, hidden_dim, num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_type, edge_pairs):
        # R-GCN forward pass
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_type))
        
        # Link prediction
        edge_embeddings = []
        for u, v in edge_pairs:
            u_emb = h[u]
            v_emb = h[v]
            edge_emb = torch.cat([u_emb, v_emb], dim=0)
            edge_embeddings.append(edge_emb)
        
        edge_embeddings = torch.stack(edge_embeddings)
        predictions = self.link_predictor(edge_embeddings)
        
        return predictions.squeeze()

class RandomBaseline:
    """ランダムベースライン"""
    def __init__(self):
        pass
    
    def predict(self, edge_pairs):
        return np.random.random(len(edge_pairs))

class BERTCosineSimilarityBaseline:
    """BERTコサイン類似度ベースライン"""
    def __init__(self, node_embeddings, node_to_idx):
        self.node_embeddings = node_embeddings
        self.node_to_idx = node_to_idx
        self.embedding_matrix = np.array([node_embeddings[node] for node in sorted(node_to_idx.keys())])
    
    def predict(self, edge_pairs):
        similarities = []
        for u, v in edge_pairs:
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            u_emb = self.embedding_matrix[u_idx]
            v_emb = self.embedding_matrix[v_idx]
            sim = cosine_similarity([u_emb], [v_emb])[0][0]
            similarities.append(sim)
        return np.array(similarities)

class TFIDFLogisticRegressionBaseline:
    """TF-IDF + ロジスティック回帰ベースライン"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = LogisticRegression()
    
    def fit(self, train_edges, all_nodes):
        # TF-IDF特徴量を作成
        node_texts = [node for node in all_nodes]
        tfidf_matrix = self.vectorizer.fit_transform(node_texts)
        
        # エッジペア特徴量を作成
        node_to_tfidf_idx = {node: i for i, node in enumerate(all_nodes)}
        
        X_train = []
        y_train = []
        
        for (u, v), label in train_edges:
            u_idx = node_to_tfidf_idx[u]
            v_idx = node_to_tfidf_idx[v]
            u_vec = tfidf_matrix[u_idx].toarray()[0]
            v_vec = tfidf_matrix[v_idx].toarray()[0]
            
            # 特徴量を結合
            features = np.concatenate([u_vec, v_vec, u_vec * v_vec])  # concat, element-wise product
            X_train.append(features)
            y_train.append(label)
        
        self.classifier.fit(X_train, y_train)
        self.tfidf_matrix = tfidf_matrix
        self.node_to_tfidf_idx = node_to_tfidf_idx
    
    def predict(self, edge_pairs):
        X_test = []
        for u, v in edge_pairs:
            u_idx = self.node_to_tfidf_idx[u]
            v_idx = self.node_to_tfidf_idx[v]
            u_vec = self.tfidf_matrix[u_idx].toarray()[0]
            v_vec = self.tfidf_matrix[v_idx].toarray()[0]
            
            features = np.concatenate([u_vec, v_vec, u_vec * v_vec])
            X_test.append(features)
        
        return self.classifier.predict_proba(X_test)[:, 1]