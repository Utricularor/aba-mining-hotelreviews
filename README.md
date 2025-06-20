# aba-mining-hotelreviews

## Link Prediction with RGCN + DistMult

## Overview
`rgcn_link_predict.py` trains and evaluates a link prediction model that combines an **RGCN (Relational Graph Convolutional Network)** encoder and a **DistMult** decoder.  
Each node name is first embedded with **Sentence-Transformers**, and the resulting node features are passed through the RGCN.  
The DistMult decoder then predicts whether an edge between two nodes should exist (binary Yes/No).

The input is a tab-separated file with three columns:
```
Assumption<TAB>Proposition<TAB>label
```
where `label` is either `Yes` (positive edge) or anything else (negative edge).

---

## Processing Pipeline
1. **Data Loading (`load_tsv_graph`)**  
   * Reads the TSV file and builds a node vocabulary from the `head` and `tail` columns.  
   * Encodes every unique node string with a `SentenceTransformer`; the resulting vectors become node features `x` in a `torch_geometric.data.Data` object.  
   * Edge indices (`edge_index`, shape 2 × E) and binary labels (`labels`, shape E) are returned together with the graph data.

2. **Model Components**
   * **RGCNEncoder** – a stack of ≥3 `RGCNConv` layers: input → hidden → output dimensions.  
   * **DistMultDecoder** – learns a relation embedding `r` and scores an edge as  
     \(score = \sum_i h_i \; r_i \; t_i\).  
   * **LinkPredictor** – wraps the encoder and decoder.

3. **Edge Split**  
   * All edges are shuffled and split into train / validation / test sets according to `val_ratio` and `test_ratio`.

4. **Training Loop**  
   * Loss: `BCEWithLogitsLoss`  
   * Optimizer: `Adam`  
   * Every `log_every` epochs the script prints the validation AUROC.

5. **Evaluation**  
   * After training, AUROC is reported on the held-out test edges.

---

## Command-line Arguments
| Argument | Default | Description |
|---|---|---|
| `--data` | (required) | Path to the TSV file (`head tail label`) |
| `--lm_model` | `all-MiniLM-L6-v2` | Sentence-Transformer model name |
| `--hidden_dim` | 256 | Hidden dimension of the RGCN |
| `--out_dim` | 256 | Output dimension of the RGCN / decoder |
| `--num_layers` | 3 | Number of RGCN layers (≥3) |
| `--epochs` | 100 | Training epochs |
| `--lr` | 1e-3 | Learning rate (Adam) |
| `--val_ratio` | 0.1 | Fraction of edges for validation |
| `--test_ratio` | 0.1 | Fraction of edges for testing |
| `--log_every` | 10 | Log interval (epochs) |

---

## Quick Start
```bash
# Example: train for 20 epochs on the sample data
python rgcn_link_predict.py \
  --data sample.tsv \
  --epochs 20 \
  --hidden_dim 64 \
  --out_dim 64 \
  --num_layers 3 \
  --val_ratio 0.2 \
  --test_ratio 0.2
```

---

## Requirements
* Python 3.10+
* PyTorch
* PyTorch Geometric (and its CUDA dependencies)
* torchmetrics
* sentence-transformers
* pandas

Install via `make`:
```bash
make
```

---

## Output
The script prints per-epoch **Loss** and **Validation AUROC**, followed by the final **Test AUROC**.

---

## Notes
* Currently the script assumes a single relation (`edge_type = 0`). By extending the TSV with a relation column and feeding it into `edge_type`, you can handle multi-relation graphs.
* For large graphs, embedding all node strings with Sentence-Transformers can be a bottleneck. Consider caching or batching embeddings when necessary.