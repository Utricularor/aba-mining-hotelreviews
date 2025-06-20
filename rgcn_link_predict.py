import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from torchmetrics import AUROC
from sentence_transformers import SentenceTransformer
import pandas as pd


def load_tsv_graph(tsv_path: str, model_name: str) -> Tuple[Data, torch.Tensor, int]:
    """Load graph from a TSV file of the form: head<TAB>tail<TAB>label(Yes/No).

    Returns
    -------
    data: torch_geometric.data.Data with fields x, edge_index, edge_type (all zeros)
    labels: Tensor of shape (E,) with 1 for "Yes" and 0 for "No"
    num_relations: int (=1)
    """
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["head", "tail", "label"])

    # Build node vocabulary
    unique_nodes = pd.unique(df[["head", "tail"]].values.ravel())
    node2idx = {n: i for i, n in enumerate(unique_nodes)}

    # Sentence-transformers encoding
    encoder = SentenceTransformer(model_name)
    with torch.no_grad():
        node_feats = torch.tensor(
            encoder.encode(list(unique_nodes), convert_to_numpy=True), dtype=torch.float
        )

    src = df["head"].map(node2idx).values
    dst = df["tail"].map(node2idx).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)  # single relation

    labels = torch.tensor(df["label"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0).values, dtype=torch.float)

    data = Data(x=node_feats, edge_index=edge_index, edge_type=edge_type)
    num_relations = 1
    return data, labels, num_relations


class RGCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, num_relations: int):
        super().__init__()
        layers = []
        layers.append(RGCNConv(in_channels, hidden_channels, num_relations))
        for _ in range(num_layers - 2):
            layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
        layers.append(RGCNConv(hidden_channels, out_channels, num_relations))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_type):
        for conv in self.layers[:-1]:
            x = F.relu(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.layers[-1](x, edge_index, edge_type)
        return x


class DistMultDecoder(nn.Module):
    def __init__(self, num_relations: int, hidden_channels: int):
        super().__init__()
        self.rel_emb = nn.Parameter(torch.empty(num_relations, hidden_channels))
        nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        head, tail = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(head * rel * tail, dim=1)  # (E,)


class LinkPredictor(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index, edge_type):
        z = self.encoder(x, edge_index, edge_type)
        return z

    def decode(self, z, edge_index, edge_type):
        return self.decoder(z, edge_index, edge_type)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, labels, num_rel = load_tsv_graph(args.data, args.lm_model)

    # Shuffle and split edge indices
    num_edges = data.edge_index.size(1)
    perm = torch.randperm(num_edges)
    val_len = max(1, int(args.val_ratio * num_edges))
    test_len = max(1, int(args.test_ratio * num_edges))
    train_idx = perm[val_len + test_len :]
    val_idx = perm[:val_len]
    test_idx = perm[val_len : val_len + test_len]

    if train_idx.numel() == 0:  # ensure at least one training edge
        train_idx = val_idx[:1]
        val_idx = val_idx[1:]
        if val_idx.numel() == 0:
            val_idx = test_idx[:1]
            test_idx = test_idx[1:]

    encoder = RGCNEncoder(
        in_channels=data.x.size(1),
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        num_layers=args.num_layers,
        num_relations=num_rel,
    )
    decoder = DistMultDecoder(num_rel, args.out_dim)
    model = LinkPredictor(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)
    labels = labels.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encoder(data.x.to(device), edge_index, edge_type)

        train_scores = model.decoder(z, edge_index[:, train_idx], edge_type[train_idx])
        train_labels = labels[train_idx]

        loss = criterion(train_scores, train_labels)
        loss.backward()
        optimizer.step()

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_scores = model.decoder(z, edge_index[:, val_idx], edge_type[val_idx])
                val_labels = labels[val_idx]
                val_auc = AUROC(task="binary")(val_scores.cpu(), val_labels.cpu())
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        z = model.encoder(data.x.to(device), edge_index, edge_type)
        test_scores = model.decoder(z, edge_index[:, test_idx], edge_type[test_idx])
        test_labels = labels[test_idx]
        test_auc = AUROC(task="binary")(test_scores.cpu(), test_labels.cpu())
    print(f"Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN + DistMult link prediction on TSV (head, tail, Yes/No) data")
    parser.add_argument("--data", type=str, required=True, help="Path to TSV file: head<TAB>tail<TAB>label(Yes/No)")
    parser.add_argument("--lm_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model for name embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3, help="Number of RGCN layers (>=3)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()
    train(args) 

# python rgcn_link_predict.py --data sample.tsv --epochs 20 --hidden_dim 64 --out_dim 64 --num_layers 3 --val_ratio 0.2 --test_ratio 0.2