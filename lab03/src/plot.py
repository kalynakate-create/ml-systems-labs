"""Task 5: accuracy vs parameters and accuracy vs latency.

Run after cost.py and train.py:  F:\\venvs\\mlsys\\Scripts\\python.exe plot.py
Reads results/cost.json + results/accuracy.json, writes results/tradeoff.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")             # save to file, no window
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

COLORS = {"MLP": "#2a6fdb", "CNN": "#e07b00", "Attention": "#2e9e5b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()
    out = Path(args.out)

    cost = json.loads((out / "cost.json").read_text())["models"]
    acc = json.loads((out / "accuracy.json").read_text())

    panels = [
        ("Parameters", lambda c: c["params"]),
        ("Latency, batch 1 (ms, median of 100)", lambda c: c["latency_ms"]["1"]),
        ("Latency, batch 128 (ms, median of 20)", lambda c: c["latency_ms"]["128"]),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.5))
    for ax, (xlabel, get_x) in zip(axes, panels):
        for name, c in cost.items():
            x, y = get_x(c), acc[name]["test_acc"] * 100
            ax.scatter(x, y, s=80, color=COLORS[name], zorder=3)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6))
        ax.set_xscale("log")
        # ticks only at the models' own values, as plain numbers (default log ticks overlap)
        xs = [get_x(c) for c in cost.values()]
        ax.set_xticks(xs, [f"{v:,.0f}" if v >= 100 else f"{v:.2f}" for v in xs])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.margins(x=0.15)
        ax.set_xlabel(xlabel + " (log scale)")
        ax.set_ylabel("Test accuracy, %")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Fashion-MNIST: accuracy vs cost")
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=150)
    print(f"saved {out / 'tradeoff.png'}")


if __name__ == "__main__":
    main()
