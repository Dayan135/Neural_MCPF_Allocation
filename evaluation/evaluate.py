"""
Evaluation and comparison experiments (mTSP goal allocation).

Offline metrics (no solver needed):
  - Per-goal accuracy: did we predict the right agent for each goal?
  - Full-assignment accuracy: did we get every goal right for the instance?
  - Cost ratio = NN expected cost / solver optimal cost

Usage:
    python evaluate.py --checkpoint ../checkpoints/best.pt
                       [--data_dir ../data] [--split test]
"""

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.network import build_model


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = build_model(
        model_type=args.get("model_type", "mlp"),
        N=args["N"],
        M=args.get("M", args["N"]),
        hidden=args["hidden"],
        num_heads=args.get("num_heads", 4),
        num_layers=args.get("num_layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def decode_assignment(P_np: np.ndarray) -> np.ndarray:
    N, M = P_np.shape
    Y_pred = np.zeros((N, M), dtype=float)
    for j in range(M):
        best_agent = P_np[:, j].argmax()
        Y_pred[best_agent, j] = 1.0
    return Y_pred


def assignment_cost(D: np.ndarray, Y: np.ndarray) -> float:
    return float((D * Y).sum())


def offline_metrics(
    model,
    D_all: np.ndarray,
    Y_all: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> dict:
    per_goal_correct = []
    full_match = []
    cost_ratios = []

    for start in range(0, len(D_all), batch_size):
        D_batch = torch.from_numpy(D_all[start : start + batch_size]).float().to(device)
        Y_batch = Y_all[start : start + batch_size]

        with torch.no_grad():
            P_batch = model(D_batch).cpu().numpy()

        D_np = D_all[start : start + batch_size]

        for i in range(len(D_np)):
            Y_pred = decode_assignment(P_batch[i])
            Y_true = Y_batch[i]

            pred_agents = P_batch[i].argmax(axis=0)
            true_agents = Y_true.argmax(axis=0)
            per_goal_correct.append((pred_agents == true_agents).mean())
            full_match.append(np.array_equal(Y_pred, Y_true))

            opt_cost = assignment_cost(D_np[i], Y_true)
            nn_cost = assignment_cost(D_np[i], Y_pred)
            ratio = (nn_cost / opt_cost) if opt_cost > 1e-9 else 1.0
            cost_ratios.append(ratio)

    return {
        "per_goal_accuracy": float(np.mean(per_goal_correct)),
        "full_assignment_accuracy": float(np.mean(full_match)),
        "mean_cost_ratio": float(np.mean(cost_ratios)),
        "max_cost_ratio": float(np.max(cost_ratios)),
        "n_samples": len(D_all),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default=os.path.join("..", "data"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    split_dir = os.path.join(args.data_dir, args.split)
    D_all = np.load(os.path.join(split_dir, "D_matrices.npy"))
    Y_all = np.load(os.path.join(split_dir, "Y_matrices.npy"))

    print(f"\n--- Offline metrics on '{args.split}' split ({len(D_all)} samples) ---")
    metrics = offline_metrics(model, D_all, Y_all, device)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
