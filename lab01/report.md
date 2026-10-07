# Lab 1: Environment and first system measurements

## 1. Goal

Set up a reproducible Python environment and train two baseline classifiers on the Breast Cancer Wisconsin dataset. Measure their system cost (training time, inference latency, model size, peak memory) and decide on which deployment platforms (Cloud, Edge, Mobile, TinyML) each model could run.

## 2. Method

- **Environment:** Python 3.11.8 in a virtual environment (`.venv`), packages pinned in `requirements.txt`. Versions are saved in `results/versions.txt`. `pyarrow` was pinned to 15.0.2 instead of 16.1.0, because `mlflow==2.14.1` requires `pyarrow<16` and the two pinned versions cannot be installed together.
- **Data:** `load_breast_cancer(return_X_y=True)`, 569 samples, 30 features. 70/30 stratified split with `random_state=42` (398 train / 171 test).
- **Seeds:** `random`, `numpy`, `torch` and `torch.cuda` seeds set to 42.
- **Models:** `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Training time:** 1 warm-up run, then the median of 5 timed runs (`time.perf_counter`).
- **Inference latency:** prediction on a single test sample, 1 warm-up, then the median of 100 runs.
- **Model size:** the model is saved with `joblib.dump` and the file size is read from disk.
- **Peak memory:** the process memory (RSS, `psutil`) is sampled every 1 ms during training and inference, and the maximum is kept.
- **Hardware:** HARDWARE_HERE

## 3. Results

### Accuracy (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| Logistic Regression | 0.9415 |
| Random Forest | 0.9357 |

### System cost (`results/system_costs.csv`)

| Model | Train time, median (s) | Inference latency, median (ms) | Model size (bytes / KB) | Peak memory, train (MB) | Peak memory, inference (MB) |
|---|---|---|---|---|---|
| Logistic Regression | 0.3188 | 0.0694 | 1055 / 1.03 | 227.2 | LR_INFER_MB |
| Random Forest | 0.1363 | 2.8784 | 290889 / 284.07 | 227.7 | RF_INFER_MB |

### Deployment fit (`results/deployment_fit.csv`)

Budgets: Cloud (memory ≥ 1 GB, latency ≤ 100 ms, size ≤ 500 MB), Edge (256–1024 MB, ≤ 50 ms, ≤ 50 MB), Mobile (64–256 MB, ≤ 20 ms, ≤ 10 MB), TinyML (≤ 256 KB, ≤ 10 ms, ≤ 100 KB).

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | yes | yes | yes | no |
| Random Forest | yes | yes | yes | no |

**Justification:**

Both models fit Cloud, Edge and Mobile, and neither fits TinyML.

- **Logistic Regression** does not fit TinyML because its peak memory (227.2 MB) is far above the 256 KB limit. The model itself is tiny (1.03 KB) and fast (0.07 ms), so only the memory breaks the budget.
- **Random Forest** does not fit TinyML for two reasons: its peak memory (227.7 MB) is far above 256 KB, and its model file (284.07 KB) is larger than the 100 KB limit. Its latency (2.88 ms) would still be within the 10 ms limit.
- Both models fit Mobile, but only just: about 228 MB of the 256 MB memory budget is used, and almost all of it is the Python process and its libraries, not the model.

## 4. Conclusions

1. **Both models are almost equally accurate.** Logistic Regression reached 0.9415 and Random Forest 0.9357 test accuracy. The difference of 0.0058 is only one wrongly classified sample out of 171 in the test set, so on this dataset the more complex model gives no accuracy benefit.

2. **Logistic Regression is much cheaper to run.** Its single-sample latency is about 41 times lower (0.07 ms vs 2.88 ms) and its model file is about 276 times smaller (1.03 KB vs 284.07 KB). Training time was the exception: Random Forest trained faster (0.14 s vs 0.32 s), probably because Logistic Regression on unscaled data needed all 1000 iterations without converging (ConvergenceWarning). Scaling the features would likely make it train much faster.

3. **The runtime, not the model, decides where a model can be deployed.** Both models need about 227 MB of memory, which is almost entirely the Python interpreter plus NumPy and scikit-learn. This is why neither fits TinyML (256 KB) and why both are close to the Mobile limit (256 MB). To run on a microcontroller, the model would have to be exported to a lightweight format (for example, C code or a TinyML runtime) instead of running inside Python.
