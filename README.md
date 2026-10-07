# ML Systems Labs

Individual lab work. Python 3.11.8, dependencies pinned in `requirements.txt`.

## Setup (Windows)

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
