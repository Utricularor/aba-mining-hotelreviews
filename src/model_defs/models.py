import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# BERT関連のインポート
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. BERT models will not work.")

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
        
        return predictions.squeeze(-1)

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


# =============================================================================
# BERT-based Models
# =============================================================================

class ImprovedBERTLinkPredictor(nn.Module):
    """
    改良版BERT Link Predictor（固定BERT + 学習可能線形層）
    
    小データセットに適した設計:
    - BERTパラメータを固定し、分類層のみ学習
    - ノードの埋め込みを別々に取得してから結合
    """
    def __init__(self, model_name='google-bert/bert-base-uncased', 
                 dropout=0.3, max_length=128, freeze_bert=True, device='cpu'):
        super(ImprovedBERTLinkPredictor, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for BERT models")
        
        self.model_name = model_name
        self.max_length = max_length
        self.freeze_bert = freeze_bert
        self.device = device
        
        # BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # BERTのパラメータを固定（推奨）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head（学習対象）
        hidden_size = self.bert.config.hidden_size  # 768
        
        # より深い分類層（BERTが固定なので複雑な変換が必要）
        self.classifier = nn.Sequential(
            # 入力: [CLS1, CLS2] -> 768*2 = 1536次元
            nn.Linear(hidden_size * 2, hidden_size),  # 1536 -> 768
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),  # 768 -> 384
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),  # 384 -> 192
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 4, 1)  # 192 -> 1
        )
        
    def forward(self, assumption_texts, proposition_texts):
        """
        Args:
            assumption_texts: List of assumption texts
            proposition_texts: List of proposition texts
        
        Returns:
            logits: Tensor of shape (batch_size,)
        """
        # Encode assumptions
        assumption_inputs = self.tokenizer(
            assumption_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        assumption_inputs = {k: v.to(self.device) for k, v in assumption_inputs.items()}
        
        # Encode propositions
        proposition_inputs = self.tokenizer(
            proposition_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        proposition_inputs = {k: v.to(self.device) for k, v in proposition_inputs.items()}
        
        # Get BERT outputs
        assumption_outputs = self.bert(**assumption_inputs)
        proposition_outputs = self.bert(**proposition_inputs)
        
        # Use CLS token embeddings
        assumption_emb = assumption_outputs.last_hidden_state[:, 0]  # [CLS] token
        proposition_emb = proposition_outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Concatenate embeddings
        combined = torch.cat([assumption_emb, proposition_emb], dim=-1)
        
        # Classification
        logits = self.classifier(combined)
        return logits.squeeze(-1)


class CrossEncoderBERTLinkPredictor(nn.Module):
    """
    Cross-Encoder方式でノードペア関係を直接学習するBERTモデル
    
    ノードペアを単一のシーケンスとして処理し、
    BERT内部のアテンション機構でペア関係を学習
    """
    def __init__(self, model_name='google-bert/bert-base-uncased', 
                 dropout=0.3, max_length=256, freeze_bert=True, device='cpu'):
        super(CrossEncoderBERTLinkPredictor, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for BERT models")
        
        self.model_name = model_name
        self.max_length = max_length
        self.freeze_bert = freeze_bert
        self.device = device
        
        # BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # BERTのパラメータを固定（推奨）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Cross-encoder用分類層（CLSトークンから直接分類）
        hidden_size = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 768 -> 384
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),  # 384 -> 192
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 4, 1)  # 192 -> 1
        )
        
    def forward(self, assumption_texts, proposition_texts):
        """
        Args:
            assumption_texts: List of assumption texts
            proposition_texts: List of proposition texts
        
        Returns:
            logits: Tensor of shape (batch_size,)
        """
        # Cross-encoder: ノードペアを単一シーケンスとして処理
        paired_texts = []
        for assumption, proposition in zip(assumption_texts, proposition_texts):
            # [CLS] assumption [SEP] proposition [SEP] 形式
            paired_text = f"{assumption} [SEP] {proposition}"
            paired_texts.append(paired_text)
        
        # トークン化
        inputs = self.tokenizer(
            paired_texts, 
            padding=True, 
            truncation=True,
            max_length=self.max_length, 
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # BERT処理
        outputs = self.bert(**inputs)
        
        # CLSトークンの表現を使用（ペア関係を学習済み）
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # 分類
        logits = self.classifier(cls_output)
        return logits.squeeze(-1)