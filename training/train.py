"""
Training loop for mTSP goal allocation models.

Usage:
    python train.py [--data_dir DIR] [--N 2] [--M 2]
                   [--model_type {mlp,deepsets,transformer}]
                   [--hidden 64] [--num_heads 4] [--num_layers 3]
                   [--epochs 100] [--batch_size 256] [--lr 1e-3]
                   [--lam 0.1] [--seed 0] [--checkpoint_dir DIR]
                   [--run_name NAME]
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.network import build_model
from model.losses import mTSP_loss


class AllocationDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        split_dir = os.path.join(data_dir, split)
        self.D = torch.from_numpy(
            np.load(os.path.join(split_dir, "D_matrices.npy"))
        ).float()
        self.Y = torch.from_numpy(
            np.load(os.path.join(split_dir, "Y_matrices.npy"))
        ).float()

    def __len__(self) -> int:
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.Y[idx]


def per_goal_accuracy(P: torch.Tensor, Y: torch.Tensor) -> float:
    pred = P.argmax(dim=1)   # (B, M)
    true = Y.argmax(dim=1)   # (B, M)
    return (pred == true).float().mean().item()


def run_epoch(model, loader, optimizer, lam, device, train: bool):
    model.train(train)
    total_loss = total_ce = total_ms = total_acc = 0.0

    with torch.set_grad_enabled(train):
        for D, Y in loader:
            D, Y = D.to(device), Y.to(device)
            P = model(D)
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

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device}  |  model={args.model_type}  N={args.N} M={args.M} "
        f"hidden={args.hidden}  lam={args.lam}"
    )

    train_ds = AllocationDataset(args.data_dir, "train")
    val_ds = AllocationDataset(args.data_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(
        args.model_type, N=args.N, M=args.M, hidden=args.hidden,
        num_heads=args.num_heads, num_layers=args.num_layers,
    ).to(device)
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
