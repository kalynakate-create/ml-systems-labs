"""Step 4: compare measured costs with the Cloud / Edge / Mobile / TinyML budgets.

Reads results/system_costs.csv, writes results/deployment_fit.csv and prints a Markdown table.
Memory check uses the peak process memory (the max of training and inference),
compared with the upper memory limit of each platform.
"""
import csv

from common import RESULTS_DIR

# Upper limits from the manual (memory in MB, latency in ms, size in KB).
BUDGETS = {
    "Cloud":  {"max_mem_mb": float("inf"), "max_latency_ms": 100, "max_size_kb": 500 * 1024},
    "Edge":   {"max_mem_mb": 1024,         "max_latency_ms": 50,  "max_size_kb": 50 * 1024},
    "Mobile": {"max_mem_mb": 256,          "max_latency_ms": 20,  "max_size_kb": 10 * 1024},
    "TinyML": {"max_mem_mb": 256 / 1024,   "max_latency_ms": 10,  "max_size_kb": 100},
}


def main() -> None:
    with open(RESULTS_DIR / "system_costs.csv") as f:
        costs = list(csv.DictReader(f))

    rows = []
    for c in costs:
        mem = max(float(c["peak_rss_train_mb"]), float(c["peak_rss_infer_mb"]))
        lat = float(c["inference_latency_median_ms"])
        size = float(c["model_size_kb"])
        for platform, b in BUDGETS.items():
            mem_ok = mem <= b["max_mem_mb"]
            lat_ok = lat <= b["max_latency_ms"]
            size_ok = size <= b["max_size_kb"]
            rows.append({
                "model": c["model"], "platform": platform,
                "memory_ok": mem_ok, "latency_ok": lat_ok, "size_ok": size_ok,
                "fits": mem_ok and lat_ok and size_ok,
            })

    with open(RESULTS_DIR / "deployment_fit.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mark = lambda ok: "yes" if ok else "no"
    print("| Model | Platform | Memory | Latency | Size | Fits? |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['platform']} | {mark(r['memory_ok'])} | "
              f"{mark(r['latency_ok'])} | {mark(r['size_ok'])} | **{mark(r['fits'])}** |")
    print(f"\nSaved {RESULTS_DIR / 'deployment_fit.csv'}")


if __name__ == "__main__":
    main()
