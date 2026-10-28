# Lab 2 — A neural network from scratch

Code: `src/lab2.py` (NumPy only; scikit-learn is used just to load and split the data).
Run `python lab02/src/lab2.py` from the repo root; all numbers below come from `results/summary.txt`.
Environment: Python 3.11.9, NumPy 2.4.2, scikit-learn 1.8.0, Matplotlib 3.10.8.

## 1. Dataset [10]

`load_digits`: 1797 images of 8×8 = 64 pixels, values 0–16, 10 classes.

- Inputs divided by 16 → range [0, 1].
- Labels one-hot encoded with `np.eye(10)[y]`.
- `train_test_split(test_size=0.3, stratify=y, random_state=42)` → train 1257 × 64, test 540 × 64.
  Stratification keeps 122–128 training and 52–55 test samples per class.

## 2. Network design [15]

```
x (batch, 64)
  → z1 = x·W1 + b1   (batch, W)   → a1 = ReLU(z1)
  → z2 = a1·W2 + b2  (batch, W)   → a2 = ReLU(z2)
  → z3 = a2·W3 + b3  (batch, 10)  → p  = softmax(z3)
```

| Parameter | Shape | W = 16 | W = 64 |
|---|---|---|---|
| W1 | (64, W) | 1024 | 4096 |
| b1 | (W,) | 16 | 64 |
| W2 | (W, W) | 256 | 4096 |
| b2 | (W,) | 16 | 64 |
| W3 | (W, 10) | 160 | 640 |
| b3 | (10,) | 10 | 10 |
| **Total** | | **1482** | **8970** |

Weights ~ N(0, 0.01), biases = 0.
**Ambiguity:** N(0, 0.01) can mean std = 0.01 or variance = 0.01 (std = 0.1).
The graded runs use std = 0.01. The std = 0.1 reading is reported as a supplementary experiment (section 6).

## 3. Forward pass [20]

Implemented layer by layer in `forward()`. Softmax subtracts the row maximum before `exp` for numerical stability.

| W | Output shape | max \|row sum − 1\| | Result |
|---|---|---|---|
| 16 | (1257, 10) | 4.4e-16 | PASS |
| 64 | (1257, 10) | 3.3e-16 | PASS |

## 4. Cross-entropy loss [10]

L = −(1/N) Σᵢ Σₖ Yᵢₖ log pᵢₖ, computed from the logits with log-sum-exp, so `log(0)` never happens.

Untrained loss: **2.3026** for both widths, equal to ln 10 = 2.3026.
Reason: with tiny weights z3 ≈ 0, so softmax gives ≈ 1/10 for every class, and −log(1/10) = ln 10.

## 5. Backpropagation and gradient check [30]

### Derivation (chain rule)

For softmax + cross-entropy, the gradient w.r.t. the logits collapses to a simple form (mean over N samples):

```
dz3 = (p − Y) / N
dW3 = a2ᵀ · dz3           db3 = Σ_rows dz3
da2 = dz3 · W3ᵀ
dz2 = da2 ⊙ 1[z2 > 0]     (ReLU' = 1 if z > 0 else 0)
dW2 = a1ᵀ · dz2           db2 = Σ_rows dz2
da1 = dz2 · W2ᵀ
dz1 = da1 ⊙ 1[z1 > 0]
dW1 = xᵀ · dz1            db1 = Σ_rows dz1
```

Each dW has the same shape as its W, e.g. a2ᵀ (W, N) · dz3 (N, 10) → (W, 10). dx is not computed because the input is not trained.

### Numerical check

Central difference: (L(w + h) − L(w − h)) / 2h with h = 1e-5.
Relative error: |g_analytic − g_numeric| / (|g_analytic| + |g_numeric|).
The check runs on one mini-batch of 32 training samples.

**Graded check (5a):** weights initialised with std 0.1; 10 random weights from W1/W2/W3 whose analytic gradient is non-zero.

| W | Passed | Worst relative error |
|---|---|---|
| 16 | **10/10 — PASS** | 8.0e-08 |
| 64 | **10/10 — PASS** | 1.1e-07 |

**Diagnostic check (5b):** spec init std 0.01, fully random weights → 8/10 (W = 16), 6/10 (W = 64).

Why the spec init fails, and why this is not a bug:

1. **Floating-point floor.** The loss is ≈ 2.3 and float64 has a relative precision of ≈ 2.2e-16. The numerical slope therefore has an absolute error of ≈ 2.2e-16 · 2.3 / 1e-5 ≈ 5e-11 or more. A relative error below 1e-6 is only reachable if |g| is well above ~5e-5. With std 0.01, gradients reaching W1 pass through two layers of tiny weights and are 1e-6–1e-7. Every failing weight in 5b has |g| ≈ 1e-6 and still agrees in the first 5–6 significant digits.
2. **ReLU kinks.** With the full training set (1257 samples), some pre-activations sit within ±h·x of zero, and the nudge flips the ReLU. Using one 32-sample batch makes this rare.
3. **Trivial passes.** Some gradients are exactly zero, for two reasons:
   - In the 32-sample batch, 13 of 64 pixels are 0 in every image (mostly the border). The W1 rows of those pixels get a zero gradient.
   - Some hidden units are inactive (ReLU = 0) for the whole batch, so the weights leaving them get a zero gradient. That is why W3 (14, 0) is 0 in 5b.

   Such a weight gives 0 = 0, which "passes" but proves nothing, so the graded check re-draws it.

An offline diagnostic sampled 300 weights in each configuration: init vs after 3 epochs, batch 1/4/32/1257. Every failure was explained by (1) or (2); none pointed to a wrong formula.
Weight values do not change whether the derivative formulas are correct, so checking at std 0.1 is a valid test of the implementation.

## 6. Training [20]

Plain SGD, mini-batch 32, 50 epochs, reshuffled every epoch, seed 0. Each epoch logs train loss, test loss and test accuracy to `results/training_log.txt`.

### Graded runs — init std 0.01 (`results/loss_curves.png`)

| W | lr | Final train loss | Final test loss | **Test accuracy** |
|---|---|---|---|---|
| 16 | 0.001 | 2.3025 | 2.3025 | 0.1019 |
| 16 | 0.01 | 2.3025 | 2.3025 | 0.1019 |
| 16 | 0.1 | 1.2317 | 1.2872 | 0.4852 |
| 64 | 0.001 | 2.3025 | 2.3025 | 0.1019 |
| 64 | 0.01 | 2.3023 | 2.3023 | 0.1019 |
| 64 | 0.1 | 0.2162 | 0.2847 | **0.9111** |

On the plot the lr = 0.001 and lr = 0.01 curves lie on top of each other at ln 10, so only one flat line is visible.

### Supplementary — init std 0.1 (`results/loss_curves_std0.1.png`)

| W | lr | Final train loss | Final test loss | **Test accuracy** |
|---|---|---|---|---|
| 16 | 0.001 | 2.2918 | 2.2915 | 0.1315 |
| 16 | 0.01 | 1.3362 | 1.3208 | 0.5648 |
| 16 | 0.1 | 0.0518 | 0.1693 | 0.9407 |
| 64 | 0.001 | 2.1972 | 2.1958 | 0.3741 |
| 64 | 0.01 | 0.2824 | 0.3111 | 0.9148 |
| 64 | 0.1 | 0.0130 | 0.1186 | **0.9630** |

### Observations

- **Init std 0.01 traps the network on a plateau.** Each layer shrinks the signal by roughly std·√fan_in/√2 (≈ 0.06 for W = 64, ≈ 0.03 for W = 16). The output is ≈ 0, and the gradient to every layer is scaled by the tiny weights of the other layers. The net sits at the saddle point around w = 0 with loss = ln 10. Accuracy 0.1019 = 55/540 means it predicts a single class.
  - lr = 0.1 escapes after ~18 epochs (W = 64) or ~30 epochs (W = 16).
  - lr = 0.01 and 0.001 never escape in 50 epochs.
- **With std 0.1 every configuration moves**, and the learning-rate ordering is clean: 0.1 > 0.01 > 0.001.
- **Wider is better** in every comparison here. W = 64 has more parameters and, at std 0.01, a larger signal gain, so it leaves the plateau sooner.
- **lr = 0.1 is noisy.** Loss spikes (e.g. epochs 28–32 at W = 64, std 0.01) are SGD steps overshooting. The best model (W = 64, lr = 0.1, std 0.1) also overfits mildly: train loss 0.013 vs test loss 0.119.

## 7. Timing [10]

Median of 100 runs after 5 warm-up calls, `time.perf_counter`. Forward = 3 matmuls + 2 ReLU + softmax. Backward = all gradients given the cached forward values.

| W | Batch | Forward, µs | Backward, µs | Backward / forward | Matmul MFLOP fwd | Matmul MFLOP bwd |
|---|---|---|---|---|---|---|
| 16 | 32 | 67.6 | 76.0 | 1.12 | 0.092 | 0.119 |
| 16 | 1257 | 804.9 | 747.1 | 0.93 | 3.620 | 4.666 |
| 64 | 32 | 102.7 | 133.5 | 1.30 | 0.565 | 0.868 |
| 64 | 1257 | 1695.6 | 2624.6 | **1.55** | 22.204 | 34.110 |

(Absolute times vary between runs and machines; the ratios are what matters.)

**Backward dominates once the matrix multiplications are large enough, and the reason is the number of matmuls.**

- Forward has 3 matrix products, one per layer: x·W1, a1·W2, a2·W3.
- Backward needs, per layer, one product for the weight gradient (aᵀ·dz) and one to send the gradient back (dz·Wᵀ). For the first layer the second product is skipped because dx is not needed. That makes 5 products, so backward has ≈ 1.3–1.54× the forward FLOPs.
- For W = 64 on the full batch, the measured ratio of 1.55 matches the FLOP ratio of 1.54: the cost is matmul-bound.
- The transposed products (aᵀ·dz sums over the batch dimension) are also less cache-friendly. A separate profile of only the matmuls: 0.65 ms forward vs 1.88 ms backward (W = 64, batch 1257).

When matrices are small, matmul FLOPs stop being the bottleneck:

- At batch 32, every call costs tens of µs of fixed NumPy overhead, so the ratio drifts towards 1.
- At W = 16, softmax (`exp`, max, sum, divide over a (1257, 10) array) costs ≈ 0.33 ms. That is as much as all three forward matmuls together (≈ 0.30 ms), so forward even comes out slightly slower than backward.

Conclusion: in a realistically sized layer the training step is dominated by matrix multiplication, and backward costs about 1.5–2× forward, because it performs about twice as many matmuls per layer.

## Graded evidence

| Evidence | Location |
|---|---|
| Passing gradient check | Section 5a, `results/summary.txt` |
| Loss curves | `results/loss_curves.png` (spec), `results/loss_curves_std0.1.png` (supplementary) |
| Timing breakdown | Section 7, `results/summary.txt` |
