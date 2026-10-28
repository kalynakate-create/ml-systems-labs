"""Task 4: train each model for 5 epochs with Adam(lr=1e-3), batch 128; record test accuracy.

Run:  F:\\venvs\\mlsys\\Scripts\\python.exe train.py
Quick smoke test:  ... train.py --epochs 1 --max-batches 20 --out <tmp dir>
Writes results/accuracy.json.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from common import MODELS, SEED, get_loaders


@torch.inference_mode()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def train_one(model, train_dl, val_dl, epochs, max_batches=None):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, n_batches = 0.0, 0
        for x, y in train_dl:
            if max_batches and n_batches >= max_batches:
                break
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        val_acc = accuracy(model, val_dl)
        history.append({"epoch": epoch, "train_loss": running / n_batches, "val_acc": val_acc,
                        "seconds": time.perf_counter() - t0})
        print(f"  epoch {epoch}: loss={history[-1]['train_loss']:.4f}  val_acc={val_acc:.4f}  "
              f"({history[-1]['seconds']:.1f} s)")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (smoke test)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()

    train_dl, val_dl, test_dl = get_loaders()
    results = {}
    for name, cls in MODELS.items():
        torch.manual_seed(SEED)        # same init seed for every model
        model = cls()
        print(f"{name}:")
        history = train_one(model, train_dl, val_dl, args.epochs, args.max_batches)
        test_acc = accuracy(model, test_dl)
        print(f"  test_acc={test_acc:.4f}\n")
        results[name] = {"test_acc": test_acc, "history": history}

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "accuracy.json").write_text(json.dumps(results, indent=2))
    print(f"saved {out / 'accuracy.json'}")


if __name__ == "__main__":
    main()
