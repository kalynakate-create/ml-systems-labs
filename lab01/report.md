# Lab 1: Environment and first system measurements

## 1. Goal

Set up a reproducible Python environment and train two baseline classifiers on the Breast Cancer Wisconsin dataset. Measure their system cost (training time, inference latency, model size, peak memory) and decide on which deployment platforms (Cloud, Edge, Mobile, TinyML) each model could run.

## 2. Method

- **Environment:** Python 3.11.8 in a virtual environment (`.venv`), packages pinned in `requirements.txt`. Versions are saved in `results/versions.txt`.
- **Data:** `load_breast_cancer(return_X_y=True)`, 569 samples, 30 features. 70/30 stratified split with `random_state=42` (398 train / 171 test).
- **Seeds:** `random`, `numpy`, `torch` and `torch.cuda` seeds set to 42.
- **Models:** `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Training time:** 1 warm-up run, then the median of 5 timed runs (`time.perf_counter`).
- **Inference latency:** prediction on a single test sample, 1 warm-up, then the median of 100 runs.
- **Model size:** the model is saved with `joblib.dump` and the file size is read from disk.
- **Peak memory:** the process memory (RSS, `psutil`) is sampled every 1 ms during training and inference, and the maximum is kept.
- **Hardware:** <!-- TODO: your CPU, RAM and operating system, e.g. "Intel Core i5-1235U, 16 GB RAM, Windows 11" -->

## 3. Results

### Accuracy (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| Logistic Regression | <!-- TODO --> |
| Random Forest | <!-- TODO --> |

### System cost (`results/system_costs.csv`)

| Model | Train time, median (s) | Inference latency, median (ms) | Model size (bytes / KB) | Peak memory, train (MB) | Peak memory, inference (MB) |
|---|---|---|---|---|---|
| Logistic Regression | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| Random Forest | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

### Deployment fit (`results/deployment_fit.csv`)

Budgets: Cloud (memory ≥ 1 GB, latency ≤ 100 ms, size ≤ 500 MB), Edge (256–1024 MB, ≤ 50 ms, ≤ 50 MB), Mobile (64–256 MB, ≤ 20 ms, ≤ 10 MB), TinyML (≤ 256 KB, ≤ 10 ms, ≤ 100 KB).

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| Random Forest | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

**Justification:**
<!-- TODO: for each model, explain each "no" with the number that breaks the budget,
     e.g. "Random Forest is 284 KB, which is above the TinyML limit of 100 KB". -->

## 4. Conclusions

<!-- TODO: write three conclusions in your own words. Questions to think about:
     - Which model is more accurate, and is the difference big?
     - Which model is cheaper in latency and size, and by how much?
     - Why does nothing fit TinyML? (Hint: how much memory does the Python process alone use?)
     - Which model would you choose for a phone app, and why? -->

1.
2.
3.
