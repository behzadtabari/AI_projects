"""
Small binomial (binary) node-classification task on synthetic graphs,
trained with a GAT. Comparing two weight initializations:

1) randn: weights ~ Normal(0, 1)  (like np.random.randn)
2) rand : weights ~ Uniform(0, 1) (like np.random.rand)  <-- non-zero mean

This script runs BOTH experiments with the same data/seed and prints
loss/accuracy curves.

Requirements:
  pip install torch torch-geometric

Run:
  python gat_init_compare.py
"""

import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.nn import GATConv



# Reproducibility
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic is slower, but helps comparisons
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# Synthetic binomial task
# -----------------------
def make_synthetic_graph(
    num_nodes=800,
    num_node_features=16,
    p_edge=0.01,
    seed=42,
):
    """
    Create a random graph and a binary label per node.
    Labels are generated from a planted linear separator + noise.
    """
    gen = torch.Generator().manual_seed(seed)

    # Graph structure
    edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=p_edge, directed=False)

    # Node features
    x = torch.randn((num_nodes, num_node_features), generator=gen)

    # Create a "true" weight vector and bias to generate logits
    w_true = torch.randn((num_node_features, 1), generator=gen)
    b_true = 0.25

    logits = (x @ w_true).squeeze(-1) + b_true
    logits = logits + 0.25 * torch.randn(num_nodes, generator=gen)  # noise

    # Binomial labels: y in {0,1}
    y = (logits > 0.0).long()

    # Train/val/test splits
    idx = torch.randperm(num_nodes, generator=gen)
    n_train = int(0.7 * num_nodes)
    n_val = int(0.15 * num_nodes)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data



# GAT model
class SmallGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        # output dim of gat1 = hidden_dim * heads
        self.gat2 = GATConv(hidden_dim * heads, 2, heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x



# Initialization experiment
def init_weights_like_numpy(model: nn.Module, mode: str, scale: float = 0.1):
    """
    mode:
      - "randn": torch.randn_like (np.random.randn)
      - "rand" : torch.rand_like  (np.random.rand)  -> mean ~ 0.5 (non-zero mean)

    We apply this to all 2D weight tensors we can find; biases are set to 0.

    NOTE: Using Uniform(0,1) is intentionally "bad-ish" because of its non-zero mean,
    to demonstrate the difference the user asked for.
    """
    assert mode in {"randn", "rand"}

    for name, param in model.named_parameters():
        if param.dim() == 2:  # weights
            with torch.no_grad():
                if mode == "randn":
                    param.copy_(torch.randn_like(param) * scale)
                else:  # "rand"
                    param.copy_(torch.rand_like(param) * scale)
        elif param.dim() == 1:  # biases (often 1D)
            with torch.no_grad():
                param.zero_()


@torch.no_grad()
def accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def train_one_run(data: Data, init_mode: str, device: str = "cpu",
                  epochs: int = 150, lr: float = 2e-3, weight_decay: float = 5e-4):
    model = SmallGAT(in_dim=data.num_node_features, hidden_dim=32, heads=4, dropout=0.2).to(device)
    data = data.to(device)

    # Apply the requested init
    init_weights_like_numpy(model, mode=init_mode, scale=0.1)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        out = model(data.x, data.edge_index)

        tr_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
        va_acc = accuracy(out[data.val_mask], data.y[data.val_mask])

        history["train_loss"].append(loss.item())
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if epoch % 25 == 0 or epoch == 1:
            print(f"[{init_mode:5s}] epoch {epoch:3d} | loss {loss.item():.4f} | "
                  f"train_acc {tr_acc:.3f} | val_acc {va_acc:.3f}")

    # Final test
    model.eval()
    out = model(data.x, data.edge_index)
    test_acc = accuracy(out[data.test_mask], data.y[data.test_mask])
    return history, test_acc


def summarize(history, test_acc, label):
    best_val = max(history["val_acc"])
    best_val_epoch = int(np.argmax(history["val_acc"])) + 1
    final_train = history["train_acc"][-1]
    final_val = history["val_acc"][-1]
    print(f"\n=== Summary: {label} ===")
    print(f"Best val_acc : {best_val:.3f} at epoch {best_val_epoch}")
    print(f"Final train  : {final_train:.3f}")
    print(f"Final val    : {final_val:.3f}")
    print(f"Test acc     : {test_acc:.3f}")


def main():
    seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    data = make_synthetic_graph(
        num_nodes=800,
        num_node_features=16,
        p_edge=0.01,
        seed=42,
    )

    print("\nRunning experiment A: randn init (Normal)")
    hist_randn, test_randn = train_one_run(data, init_mode="randn", device=device)

    print("\nRunning experiment B: rand init (Uniform(0,1))")
    hist_rand, test_rand = train_one_run(data, init_mode="rand", device=device)

    summarize(hist_randn, test_randn, "randn (Normal)")
    summarize(hist_rand, test_rand, "rand (Uniform(0,1))")

    # Quick side note printed so you remember what is being compared
    print("\nNote:")
    print("- randn is zero-mean; rand (0..1) has mean ~0.5, which often hurts optimization.")
    print("- If you want a fairer uniform baseline, try uniform(-a, a) (zero-mean) instead.")


if __name__ == "__main__":
    main()
