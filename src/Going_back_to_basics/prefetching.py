"""
This script benchmarks the effect of DataLoader prefetching and num_workers on training speed. It uses a simple synthetic graph regression task and a small GNN.

Requirements:
  pip install torch torch-geometric pandas matplotlib networkx

Run:
  python prefetching.py
"""
import time
import math
from networkx import display
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


# Synthetic dataset (on-the-fly) 
class SyntheticGraphDataset(torch.utils.data.Dataset):
    """
    On-the-fly graph generation. Increase cpu_work to make CPU a bottleneck
    so num_workers/prefetch settings become visible.
    """
    def __init__(self, length=6000, num_nodes=64, feat_dim=16, cpu_work=0):
        self.length = length
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.cpu_work = cpu_work

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n = self.num_nodes
        f = self.feat_dim

        x = torch.randn(n, f)

        E = 4 * n
        src = torch.randint(0, n, (E,))
        dst = torch.randint(0, n, (E,))
        edge_index = torch.stack([src, dst], dim=0)

        y = x.mean().unsqueeze(0) + 0.05 * torch.randn(1)

        # Simulated CPU preprocessing (optional)
        # This is purposely simple so it runs everywhere.
        # Increase cpu_work to emphasize CPU pipeline effects.
        for _ in range(self.cpu_work):
            x = x * 1.000001 + 0.000001

        return Data(x=x, edge_index=edge_index, y=y)


# Simple graph-level regressor
class GraphRegressor(nn.Module):
    def __init__(self, in_dim=16, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)   # [num_graphs, hidden]
        return self.mlp(g).view(-1)      # [num_graphs]


# Optional CUDA stream prefetcher 
class CUDAPrefetcher:
    """
    Prefetch next batch to GPU on a separate CUDA stream (CUDA only).
    If device is CPU, it becomes a plain iterator.
    """
    def __init__(self, loader, device):
        self.it = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.next_batch = None
            return

        if self.device.type == "cuda":
            with torch.cuda.stream(self.stream):
                self.next_batch = batch.to(self.device, non_blocking=True)
        else:
            self.next_batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch


def benchmark_one(
    device,
    dataset,
    batch_size=64,
    steps=300,
    warmup=30,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=2,
    use_cuda_prefetcher=False,
    lr=1e-3,
    seed=0
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Build loader
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)

    # Model + optimizer
    model = GraphRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Iterate
    model.train()
    step_times = []
    graphs_processed = 0

    # Choose iterator
    if use_cuda_prefetcher and device.type == "cuda":
        iterator = CUDAPrefetcher(loader, device)
    else:
        iterator = iter(loader)

    # We'll cycle the loader until we hit warmup+steps steps
    target_total = warmup + steps
    i = 0
    while i < target_total:
        try:
            batch = next(iterator)
        except StopIteration:
            # Recreate iterator
            if use_cuda_prefetcher and device.type == "cuda":
                iterator = CUDAPrefetcher(loader, device)
            else:
                iterator = iter(loader)
            batch = next(iterator)

        # If not using CUDA prefetcher, move batch here
        if not (use_cuda_prefetcher and device.type == "cuda"):
            if device.type == "cuda":
                batch = batch.to(device, non_blocking=True)
            else:
                batch = batch.to(device)

        sync()
        t0 = time.perf_counter()

        pred = model(batch)
        y = batch.y.view(-1).to(device)
        loss = F.mse_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        sync()
        t1 = time.perf_counter()

        if i >= warmup:
            step_times.append(t1 - t0)
            graphs_processed += batch.num_graphs

        i += 1

    avg_step = sum(step_times) / len(step_times)
    ms_per_step = avg_step * 1000.0
    graphs_per_sec = graphs_processed / sum(step_times)

    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "cuda_prefetcher": bool(use_cuda_prefetcher and device.type == "cuda"),
        "ms_per_step": ms_per_step,
        "graphs_per_sec": graphs_per_sec,
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Make CPU more visible by increasing cpu_work (try 0, 50, 200)
dataset = SyntheticGraphDataset(length=8000, num_nodes=64, feat_dim=16, cpu_work=80)

batch_size = 64
steps = 250
warmup = 30

# A small set of configs that typically shows differences in Colab
configs = [
    # Baseline
    dict(num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=2, use_cuda_prefetcher=False),

    # Workers only (CPU prefetching)
    dict(num_workers=2, pin_memory=False, persistent_workers=True,  prefetch_factor=2, use_cuda_prefetcher=False),
    dict(num_workers=4, pin_memory=False, persistent_workers=True,  prefetch_factor=2, use_cuda_prefetcher=False),

    # Workers + pinned memory (helps H2D on CUDA)
    dict(num_workers=2, pin_memory=True,  persistent_workers=True,  prefetch_factor=2, use_cuda_prefetcher=False),
    dict(num_workers=4, pin_memory=True,  persistent_workers=True,  prefetch_factor=2, use_cuda_prefetcher=False),

    # Workers + pinned memory + CUDA stream prefetcher (CUDA only)
    dict(num_workers=2, pin_memory=True,  persistent_workers=True,  prefetch_factor=2, use_cuda_prefetcher=True),
    dict(num_workers=4, pin_memory=True,  persistent_workers=True,  prefetch_factor=2, use_cuda_prefetcher=True),

    # Try larger prefetch_factor (can help or hurt depending on RAM / variability)
    dict(num_workers=4, pin_memory=True,  persistent_workers=True,  prefetch_factor=4, use_cuda_prefetcher=True),
]

results = []
for cfg in configs:
    r = benchmark_one(
        device=device,
        dataset=dataset,
        batch_size=batch_size,
        steps=steps,
        warmup=warmup,
        **cfg
    )
    results.append(r)
    print(r)

df = pd.DataFrame(results)

# Make a readable label for plots
def make_label(row):
    pf = row["prefetch_factor"]
    pf_str = f"pf={pf}" if pf is not None else "pf=-"
    return f"w={row['num_workers']} pin={int(row['pin_memory'])} pers={int(row['persistent_workers'])} {pf_str} cudaPref={int(row['cuda_prefetcher'])}"

df["label"] = df.apply(make_label, axis=1)

    
# Plot graphs/sec
plt.figure()
plt.xticks(rotation=75, ha="right")
plt.plot(df["label"], df["graphs_per_sec"], marker="o")
plt.ylabel("graphs/sec")
plt.title("Throughput vs DataLoader settings")
plt.tight_layout()
plt.show()

# --- Plot ms/step ---
plt.figure()
plt.xticks(rotation=75, ha="right")
plt.plot(df["label"], df["ms_per_step"], marker="o")
plt.ylabel("ms/step")
plt.title("Step time vs DataLoader settings")
plt.tight_layout()
plt.show()