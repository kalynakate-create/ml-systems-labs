# ML Systems Labs

Individual lab work. Each lab lives in `labNN/`: code in `src/`, outputs in `results/`, write-up in
`report.md`. All commands below are run from the repo root.

## Setup (Windows)

Lab 1 uses the root `requirements.txt` (Python 3.11.8). Labs 2–4 were run with newer packages, pinned
in their own `labNN/requirements.txt`.

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lab 1

```
python lab01/src/print_versions.py > lab01/results/versions.txt
python lab01/src/train_baselines.py
python lab01/src/measure_cost.py
python lab01/src/deployment_fit.py
```

Results are saved in `lab01/results/`, the write-up is in `lab01/report.md`.

## Lab 2 — neural network from scratch

```
pip install -r lab02/requirements.txt
python lab02/src/lab2.py
```

NumPy two-hidden-layer network on `load_digits`: forward pass, backprop, gradient check, SGD,
timing. Outputs: loss curves, `summary.txt`, `training_log.txt` in `lab02/results/`.

## Lab 3 — architectures and their costs

```
pip install -r lab03/requirements.txt
python lab03/src/cost.py
python lab03/src/train.py
python lab03/src/plot.py
```

MLP / CNN / attention on Fashion-MNIST: params, FLOPs, latency (`cost.py`, ~1 min), 5-epoch training
(`train.py`, several minutes on CPU), accuracy-vs-cost plot (`plot.py`). Dataset is downloaded to
`F:\datasets` (`DATA_ROOT` in `lab03/src/common.py`).

## Lab 4 — data pipeline and quality

```
pip install -r lab04/requirements.txt
python lab04/src/run.py
```

One command runs all steps on UCI Adult: ingest → validate → transform → features → label audit →
finalize. Parameters in `lab04/config.yaml`; processed-data SHA256 and config in
`lab04/results/manifest.yaml`; data card in `lab04/data_card.md`. Raw and processed data go to
`lab04/data/` (not committed, rebuilt by `run.py`).
