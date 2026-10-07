"""Step 3: measure training time, single-sample latency, model size and peak memory.

Output: results/system_costs.csv, results/model_<name>.joblib, results/model.joblib
"""
import csv
import shutil
import statistics
import threading
import time
from pathlib import Path

import joblib
import psutil

from common import RESULTS_DIR, load_split, make_models, set_seeds

N_TRAIN_RUNS = 5        # median of 5 runs after 1 warm-up
N_INFER_RUNS = 100      # single-sample inference repeated 100 times
PROCESS = psutil.Process()


def rss_mb() -> float:
    return PROCESS.memory_info().rss / 1024 ** 2


class PeakMemory:
    """Samples the process RSS in a background thread and keeps the maximum (in MB)."""

    def __init__(self, interval_s: float = 0.001):
        self.interval_s = interval_s
        self.peak = 0.0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, rss_mb())
            time.sleep(self.interval_s)

    def __enter__(self):
        self.start_mb = rss_mb()
        self.peak = self.start_mb
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, rss_mb())


def measure(name, make_model, X_train, y_train, X_test):
    # --- Training time: 1 warm-up, then median of 5 ---
    make_model().fit(X_train, y_train)                       # warm-up (not timed)
    train_times = []
    for _ in range(N_TRAIN_RUNS):
        model = make_model()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)
    train_s = statistics.median(train_times)

    # --- Peak memory during training ---
    with PeakMemory() as mem_train:
        model = make_model()
        model.fit(X_train, y_train)

    # --- Single-sample inference: 1 warm-up, then median of 100 ---
    sample = X_test[:1]                                      # shape (1, 30)
    model.predict(sample)                                    # warm-up
    infer_times = []
    with PeakMemory() as mem_infer:
        for _ in range(N_INFER_RUNS):
            t0 = time.perf_counter()
            model.predict(sample)
            infer_times.append(time.perf_counter() - t0)
    latency_ms = statistics.median(infer_times) * 1000

    # --- Model size on disk ---
    path = RESULTS_DIR / f"model_{name}.joblib"
    joblib.dump(model, path)
    size_bytes = path.stat().st_size

    return {
        "model": name,
        "train_time_median_s": round(train_s, 4),
        "inference_latency_median_ms": round(latency_ms, 4),
        "model_size_bytes": size_bytes,
        "model_size_kb": round(size_bytes / 1024, 2),
        "peak_rss_train_mb": round(mem_train.peak, 1),
        "peak_rss_infer_mb": round(mem_infer.peak, 1),
        "train_mem_increase_mb": round(mem_train.peak - mem_train.start_mb, 2),
    }, path


def main() -> None:
    set_seeds()
    X_train, X_test, y_train, _ = load_split()

    rows = []
    for name, model in make_models().items():
        print(f"Measuring {name} ...")
        row, path = measure(name, lambda m=model: type(m)(**m.get_params()), X_train, y_train, X_test)
        rows.append(row)
        for k, v in row.items():
            print(f"  {k}: {v}")

    # The manual names one file results/model.joblib: keep a copy of the Random Forest there.
    shutil.copy(RESULTS_DIR / "model_RandomForest.joblib", RESULTS_DIR / "model.joblib")

    out = RESULTS_DIR / "system_costs.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
