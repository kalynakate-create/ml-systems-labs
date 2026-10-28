import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIDTHS = [16, 64]
LEARNING_RATES = [0.001, 0.01, 0.1]
EPOCHS = 50
BATCH_SIZE = 32
INIT_STD = 0.01
ALT_STD = 0.1


def load_data():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Y = np.eye(10)[y]
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.3, stratify=y, random_state=42
    )
    return X_tr, X_te, Y_tr, Y_te


def init_params(width, rng, std=INIT_STD):
    return {
        "W1": rng.normal(0.0, std, (64, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0.0, std, (width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0.0, std, (width, 10)),
        "b3": np.zeros(10),
    }


def relu(z):
    return np.maximum(z, 0.0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    probs = softmax(z3)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "probs": probs}
    return probs, cache


def cross_entropy(z3, Y):
    m = z3.max(axis=1, keepdims=True)
    log_probs = z3 - m - np.log(np.exp(z3 - m).sum(axis=1, keepdims=True))
    return -np.mean(np.sum(Y * log_probs, axis=1))


def loss_fn(params, X, Y):
    _, cache = forward(params, X)
    return cross_entropy(cache["z3"], Y)


def backward(params, cache, Y):
    n = Y.shape[0]
    dz3 = (cache["probs"] - Y) / n
    dW3 = cache["a2"].T @ dz3
    db3 = dz3.sum(axis=0)

    da2 = dz3 @ params["W3"].T
    dz2 = da2 * (cache["z2"] > 0)
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def gradient_check(params, X, Y, rng, n_checks=10, h=1e-5, tol=1e-6, skip_zero=False):
    _, cache = forward(params, X)
    grads = backward(params, cache, Y)
    rows = []
    while len(rows) < n_checks:
        name = str(rng.choice(["W1", "W2", "W3"]))
        idx = tuple(int(rng.integers(0, s)) for s in params[name].shape)
        if skip_zero and grads[name][idx] == 0:
            continue
        original = params[name][idx]

        params[name][idx] = original + h
        loss_plus = loss_fn(params, X, Y)
        params[name][idx] = original - h
        loss_minus = loss_fn(params, X, Y)
        params[name][idx] = original

        numeric = (loss_plus - loss_minus) / (2 * h)
        analytic = grads[name][idx]
        denom = abs(analytic) + abs(numeric)
        rel_err = abs(analytic - numeric) / denom if denom > 0 else 0.0
        rows.append((name, idx, analytic, numeric, rel_err, rel_err < tol))
    return rows


def accuracy(params, X, Y):
    probs, _ = forward(params, X)
    return np.mean(probs.argmax(axis=1) == Y.argmax(axis=1))


def train(width, lr, data, std=INIT_STD, seed=0, log=None):
    X_tr, X_te, Y_tr, Y_te = data
    rng = np.random.default_rng(seed)
    params = init_params(width, rng, std)
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    n = X_tr.shape[0]

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            _, cache = forward(params, X_tr[batch])
            grads = backward(params, cache, Y_tr[batch])
            for key in params:
                params[key] -= lr * grads[key]

        tr_loss = loss_fn(params, X_tr, Y_tr)
        te_loss = loss_fn(params, X_te, Y_te)
        te_acc = accuracy(params, X_te, Y_te)
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        line = (f"std={std:<5g} W={width:<3d} lr={lr:<6g} epoch {epoch:2d}/{EPOCHS}  "
                f"train_loss={tr_loss:.4f}  test_loss={te_loss:.4f}  test_acc={te_acc:.4f}")
        if log is not None:
            log.write(line + "\n")
    return params, history


def plot_loss_curves(histories, path, title):
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(12, 4.5), sharey=True)
    epochs = np.arange(1, EPOCHS + 1)
    colors = {0.001: "#1f77b4", 0.01: "#ff7f0e", 0.1: "#2ca02c"}
    for ax, width in zip(axes, WIDTHS):
        for lr in LEARNING_RATES:
            h = histories[(width, lr)]
            ax.plot(epochs, h["train_loss"], color=colors[lr], label=f"lr={lr} train")
            ax.plot(epochs, h["test_loss"], color=colors[lr], ls="--", label=f"lr={lr} test")
        ax.axhline(np.log(10), color="grey", lw=0.8, ls=":", label="ln(10)")
        ax.set_title(f"Hidden width W = {width}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[-1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def median_time(fn, runs=100, warmup=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def matmul_flops(n, width):
    fwd = 2 * n * (64 * width + width * width + width * 10)
    bwd = fwd + 2 * n * (width * 10 + width * width)
    return fwd, bwd


def timing(data, rng):
    X_tr = data[0]
    Y_tr = data[2]
    rows = []
    for width in WIDTHS:
        params = init_params(width, rng)
        for n in [BATCH_SIZE, X_tr.shape[0]]:
            X, Y = X_tr[:n], Y_tr[:n]
            _, cache = forward(params, X)
            t_fwd = median_time(lambda: forward(params, X))
            t_bwd = median_time(lambda: backward(params, cache, Y))
            f_fwd, f_bwd = matmul_flops(n, width)
            rows.append((width, n, t_fwd, t_bwd, f_fwd, f_bwd))
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    data = load_data()
    X_tr, X_te, Y_tr, Y_te = data
    summary = []

    summary.append("== 1. Dataset ==")
    summary.append(f"train X {X_tr.shape}, Y {Y_tr.shape}; test X {X_te.shape}, Y {Y_te.shape}")
    summary.append(f"X range [{X_tr.min():.1f}, {X_tr.max():.1f}]")
    summary.append("train class counts: " + str(Y_tr.sum(axis=0).astype(int).tolist()))
    summary.append("test  class counts: " + str(Y_te.sum(axis=0).astype(int).tolist()))

    summary.append("\n== 2-4. Shapes, forward pass, untrained loss ==")
    for width in WIDTHS:
        params = init_params(width, np.random.default_rng(0))
        shapes = {k: v.shape for k, v in params.items()}
        n_params = sum(v.size for v in params.values())
        probs, cache = forward(params, X_tr)
        row_sum_err = np.abs(probs.sum(axis=1) - 1).max()
        loss0 = cross_entropy(cache["z3"], Y_tr)
        summary.append(f"W={width}: shapes {shapes}, total params {n_params}")
        summary.append(f"  output shape {probs.shape}, max |row sum - 1| = {row_sum_err:.2e} "
                       f"-> {'PASS' if probs.shape == (X_tr.shape[0], 10) and row_sum_err < 1e-6 else 'FAIL'}")
        summary.append(f"  untrained loss = {loss0:.4f} (ln 10 = {np.log(10):.4f})")

    X_gc, Y_gc = X_tr[:BATCH_SIZE], Y_tr[:BATCH_SIZE]
    checks = [
        ("5a. GRADED: init std 0.1, 10 random weights with non-zero gradient", ALT_STD, True),
        ("5b. DIAGNOSTIC: spec init std 0.01, 10 fully random weights", INIT_STD, False),
    ]
    summary.append("\n== 5. Gradient check (batch 32, h = 1e-5, pass if rel. error < 1e-6) ==")
    for title, std, skip_zero in checks:
        summary.append(f"-- {title}")
        for width in WIDTHS:
            params = init_params(width, np.random.default_rng(0), std)
            rows = gradient_check(params, X_gc, Y_gc, np.random.default_rng(123), skip_zero=skip_zero)
            passed = sum(r[5] for r in rows)
            summary.append(f"W={width}:")
            summary.append(f"  {'param':<5} {'index':<9} {'analytic':>14} {'numerical':>14} {'rel.err':>10}")
            for name, idx, a, nm, err, ok in rows:
                summary.append(f"  {name:<5} {str(idx):<9} {a:>14.6e} {nm:>14.6e} {err:>10.2e} "
                               f"{'ok' if ok else 'FAIL'}")
            summary.append(f"  passed {passed}/10 -> {'PASS' if passed >= 9 else 'FAIL'}")

    runs = [
        (INIT_STD, "loss_curves.png", "init N(0, 0.01) as std = 0.01 (spec)"),
        (ALT_STD, "loss_curves_std0.1.png", "init N(0, 0.01) as variance, std = 0.1 (supplementary)"),
    ]
    summary.append("\n== 6. Training (SGD, batch 32, 50 epochs) ==")
    with open(RESULTS / "training_log.txt", "w", encoding="utf-8") as log:
        for std, filename, label in runs:
            summary.append(f"-- {label}")
            histories = {}
            for width in WIDTHS:
                for lr in LEARNING_RATES:
                    t0 = time.perf_counter()
                    _, hist = train(width, lr, data, std=std, seed=0, log=log)
                    histories[(width, lr)] = hist
                    summary.append(f"W={width:<3d} lr={lr:<6g} final train loss {hist['train_loss'][-1]:.4f}  "
                                   f"test loss {hist['test_loss'][-1]:.4f}  "
                                   f"test acc {hist['test_acc'][-1]:.4f}  ({time.perf_counter() - t0:.1f}s)")
            plot_loss_curves(histories, RESULTS / filename,
                             f"Loss curves: SGD, batch 32, 50 epochs, {label}")
            summary.append(f"loss curves -> results/{filename}")
    summary.append("per-epoch log -> results/training_log.txt")

    summary.append("\n== 7. Timing (median of 100 runs) ==")
    summary.append(f"  {'W':>3} {'batch':>6} {'forward, us':>12} {'backward, us':>13} "
                   f"{'bwd/fwd':>8} {'MFLOP fwd':>10} {'MFLOP bwd':>10}")
    for width, n, t_f, t_b, f_f, f_b in timing(data, np.random.default_rng(0)):
        summary.append(f"  {width:>3} {n:>6} {t_f * 1e6:>12.1f} {t_b * 1e6:>13.1f} "
                       f"{t_b / t_f:>8.2f} {f_f / 1e6:>10.3f} {f_b / 1e6:>10.3f}")

    text = "\n".join(summary)
    (RESULTS / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
