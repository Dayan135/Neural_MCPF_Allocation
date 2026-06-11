"""
Training loop for mTSP goal allocation models.

Single-size usage:
    python train.py [--data_dir DIR] [--N 2] [--M 2]
                   [--model_type {mlp,deepsets,transformer}]
                   [--hidden 64] [--num_heads 4] [--num_layers 3]
                   [--epochs 100] [--batch_size 256] [--lr 1e-3]
                   [--lam 0.1] [--seed 0] [--checkpoint_dir DIR]
                   [--run_name NAME]

Mixed-size (universal) usage:
    python train.py --mixed --use_goal_dists
                   --data_dirs DIR1,DIR2,...
                   [--model_type transformer]
                   [same hyperparams as above]
"""

import argparse
import math
import os
import random as _random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.network import build_model
from model.losses import mTSP_loss


class AllocationDataset(Dataset):
    def __init__(self, data_dir: str, split: str, use_goal_dists: bool = False):
        split_dir = os.path.join(data_dir, split)
        self.D = torch.from_numpy(
            np.load(os.path.join(split_dir, "D_matrices.npy"))
        ).float()
        self.Y = torch.from_numpy(
            np.load(os.path.join(split_dir, "Y_matrices.npy"))
        ).float()
        self.G = None
        if use_goal_dists:
            g_path = os.path.join(split_dir, "G_matrices.npy")
            if not os.path.exists(g_path):
                raise FileNotFoundError(
                    f"G_matrices.npy not found at {g_path}; "
                    "regenerate data (build_dataset.py now emits it automatically)"
                )
            self.G = torch.from_numpy(np.load(g_path)).float()

    def __len__(self) -> int:
        return len(self.D)

    def __getitem__(self, idx):
        if self.G is not None:
            return self.D[idx], self.G[idx], self.Y[idx]
        return self.D[idx], self.Y[idx]


class MixedAllocationDataset(Dataset):
    """
    Loads multiple per-(N,M) datasets from a list of data directories and
    presents them as a single flat dataset.  Always expects G_matrices.npy
    (mixed training requires --use_goal_dists).

    config_indices maps (N, M) → list of flat sample indices so that
    MixedSizeBatchSampler can build shape-homogeneous batches.
    """

    def __init__(self, data_dirs: list[str], split: str):
        self.samples: list[tuple] = []
        self.config_indices: dict[tuple[int, int], list[int]] = {}

        for d in data_dirs:
            split_dir = os.path.join(d, split)
            D = torch.from_numpy(
                np.load(os.path.join(split_dir, "D_matrices.npy"))
            ).float()
            G = torch.from_numpy(
                np.load(os.path.join(split_dir, "G_matrices.npy"))
            ).float()
            Y = torch.from_numpy(
                np.load(os.path.join(split_dir, "Y_matrices.npy"))
            ).float()
            N, M = D.shape[1], D.shape[2]
            key = (N, M)
            start = len(self.samples)
            for i in range(len(D)):
                self.samples.append((D[i], G[i], Y[i]))
            if key not in self.config_indices:
                self.config_indices[key] = []
            self.config_indices[key].extend(range(start, len(self.samples)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MixedSizeBatchSampler(Sampler):
    """
    Yields batches where every index within a batch belongs to the same (N, M)
    configuration.  This lets the default DataLoader collate tensors normally
    (all same shape) without any padding or masking.

    Each epoch interleaves complete single-shape batches from all configs.
    """

    def __init__(
        self,
        dataset: MixedAllocationDataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.groups = dataset.config_indices
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        batches = []
        for indices in self.groups.values():
            idx_list = list(indices)
            if self.shuffle:
                _random.shuffle(idx_list)
            for i in range(0, len(idx_list), self.batch_size):
                chunk = idx_list[i : i + self.batch_size]
                if chunk:
                    batches.append(chunk)
        if self.shuffle:
            _random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum(
            math.ceil(len(v) / self.batch_size) for v in self.groups.values()
        )


def per_goal_accuracy(P: torch.Tensor, Y: torch.Tensor) -> float:
    pred = P.argmax(dim=1)   # (B, M)
    true = Y.argmax(dim=1)   # (B, M)
    return (pred == true).float().mean().item()


def run_epoch(model, loader, optimizer, lam, device, train: bool):
    model.train(train)
    total_loss = total_ce = total_ms = total_acc = 0.0

    with torch.set_grad_enabled(train):
        for batch in loader:
            if len(batch) == 3:
                D, G, Y = batch
                G = G.to(device)
            else:
                D, Y = batch
                G = None
            D, Y = D.to(device), Y.to(device)
            P = model(D, G=G) if G is not None else model(D)
            loss, l_ce, l_ms = mTSP_loss(P, Y, D, lam=lam)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_ce += l_ce.item()
            total_ms += l_ms.item()
            total_acc += per_goal_accuracy(P, Y)

    n = len(loader)
    return total_loss / n, total_ce / n, total_ms / n, total_acc / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join("..", "data"))
    parser.add_argument("--N", type=int, default=2, help="number of agents")
    parser.add_argument("--M", type=int, default=None, help="number of goals (defaults to N)")
    parser.add_argument("--model_type", choices=["mlp", "deepsets", "transformer"],
                        default="mlp")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4,
                        help="attention heads (transformer only)")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="transformer blocks (transformer only)")
    parser.add_argument("--use_goal_dists", action="store_true",
                        help="load G_matrices.npy and inject goal-goal context into transformer")
    parser.add_argument("--mixed", action="store_true",
                        help="train on multiple (N,M) configs with GoalAllocTransformerUniversal")
    parser.add_argument("--data_dirs", type=str, default=None,
                        help="comma-separated data dirs for --mixed mode")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_dir", default=os.path.join("..", "checkpoints"))
    parser.add_argument("--run_name", default=None,
                        help="subdirectory under checkpoint_dir for this run")
    args = parser.parse_args()

    if args.M is None:
        args.M = args.N

    if args.mixed and not args.data_dirs:
        raise ValueError("--mixed requires --data_dirs")
    if args.mixed and not args.use_goal_dists:
        raise ValueError("--mixed requires --use_goal_dists (universal model always uses G)")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mixed:
        data_dirs = [d.strip() for d in args.data_dirs.split(",")]
        print(
            f"Device: {device}  |  model=transformer(universal)  "
            f"configs={len(data_dirs)}  hidden={args.hidden}  lam={args.lam}"
        )
        train_ds = MixedAllocationDataset(data_dirs, "train")
        val_ds = MixedAllocationDataset(data_dirs, "val")
        train_loader = DataLoader(
            train_ds,
            batch_sampler=MixedSizeBatchSampler(train_ds, args.batch_size, shuffle=True),
            num_workers=2,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=MixedSizeBatchSampler(val_ds, args.batch_size, shuffle=False),
            num_workers=2,
        )
        model = build_model(
            "transformer", N=None, M=None, hidden=args.hidden,
            num_heads=args.num_heads, num_layers=args.num_layers,
            use_goal_dists=True, universal=True,
        ).to(device)
        # mark checkpoint as universal so evaluate.py can rebuild correctly
        args.universal = True
    else:
        print(
            f"Device: {device}  |  model={args.model_type}  N={args.N} M={args.M} "
            f"hidden={args.hidden}  lam={args.lam}"
        )
        train_ds = AllocationDataset(args.data_dir, "train", use_goal_dists=args.use_goal_dists)
        val_ds = AllocationDataset(args.data_dir, "val", use_goal_dists=args.use_goal_dists)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        model = build_model(
            args.model_type, N=args.N, M=args.M, hidden=args.hidden,
            num_heads=args.num_heads, num_layers=args.num_layers,
            use_goal_dists=args.use_goal_dists,
        ).to(device)
        args.universal = False
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    ckpt_dir = os.path.join(args.checkpoint_dir, args.run_name) if args.run_name else args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ce, tr_ms, tr_acc = run_epoch(
            model, train_loader, optimizer, args.lam, device, train=True
        )
        val_loss, val_ce, val_ms, val_acc = run_epoch(
            model, val_loader, optimizer, args.lam, device, train=False
        )
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} (ce {tr_ce:.4f} ms {tr_ms:.4f}) acc {tr_acc:.3f} | "
            f"val loss {val_loss:.4f} (ce {val_ce:.4f} ms {val_ms:.4f}) acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "args": vars(args)},
                os.path.join(ckpt_dir, "best.pt"),
            )

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
