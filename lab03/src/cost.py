"""Task 3: parameters (torchinfo), FLOPs (thop), inference latency.

Run:  F:\\venvs\\mlsys\\Scripts\\python.exe cost.py
Writes results/cost.json.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import thop
import torch
import torchinfo

from common import MODELS, SEED

WARMUP = 10
RUNS = {1: 100, 128: 20}          # batch size -> number of timed runs (manual: median of 100 and 20)


@torch.inference_mode()
def latency_ms(model, batch_size, runs):
    x = torch.randn(batch_size, 1, 28, 28)
    for _ in range(WARMUP):       # first calls allocate memory and pick kernels - don't time them
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()

    torch.manual_seed(SEED)
    results = {"torch": torch.__version__, "threads": torch.get_num_threads(), "models": {}}
    print(f"torch {torch.__version__}, CPU threads: {results['threads']}\n")

    for name, cls in MODELS.items():
        model = cls().eval()
        params = torchinfo.summary(model, input_size=(1, 1, 28, 28), verbose=0).total_params
        # thop returns MACs (multiply-accumulate), one MAC = 2 FLOPs
        macs, _ = thop.profile(model, inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
        row = {
            "params": params,
            "thop_macs": int(macs),
            "thop_flops": int(2 * macs),
            "latency_ms": {str(b): latency_ms(model, b, n) for b, n in RUNS.items()},
        }
        results["models"][name] = row
        lat = row["latency_ms"]
        print(f"{name:10s} params={params:>9,}  thop FLOPs={row['thop_flops']:>11,}  "
              f"latency b1={lat['1']:.3f} ms  b128={lat['128']:.3f} ms")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "cost.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out / 'cost.json'}")


if __name__ == "__main__":
    main()
