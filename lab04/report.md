# Lab 4: Data pipeline and quality

## Setup

- Software: Windows 11, Python 3.11, pandas 3.1.0, numpy 2.4.6, scikit-learn 1.9.1, joblib 1.6.0,
  PyYAML 6.0.3
- Data: UCI Adult via `fetch_openml(name='adult', version=2, as_frame=True)`, OpenML id 1590
- Code: `config.yaml` (every parameter); in `src/`: `common.py` (loading, clipper, derived features),
  one file per step: `ingest.py`, `validate.py`, `transform.py`, `features.py`, `label_audit.py`,
  `finalize.py`. `run.py` runs them in order.
- Raw numbers: `results/*.json`, run log `results/pipeline.log`, hash + config `results/manifest.yaml`

## 1. Problem definition

**Task.** Binary classification on the UCI Adult (1994 US Census) extract, loaded with
`fetch_openml(name='adult', version=2, as_frame=True)` (OpenML id 1590). Predict whether a person's
annual income is above 50,000 USD.

**Inputs.** 14 features per person: 6 numeric (`age`, `fnlwgt`, `education-num`, `capital-gain`,
`capital-loss`, `hours-per-week`) and 8 categorical (`workclass`, `education`, `marital-status`,
`occupation`, `relationship`, `race`, `sex`, `native-country`). Three categorical columns have missing
values (`workclass`, `occupation`, `native-country`, 857–2809 rows each).

**Output.** `income ∈ {<=50K, >50K}`; the positive class is `>50K`. Classes are imbalanced: 23.9 %
positive, so a model that always says `<=50K` already gets 76 % accuracy.

**Success metrics.** F1 of the positive class is the primary metric, because accuracy rewards the
majority class; accuracy is reported alongside. Both are measured once, on a held-out 30 % stratified
test split that is never used for fitting anything.

**Constraints.**

- *No leakage.* The split happens before any statistic is computed. Imputation values, clip
  percentiles, scaling and one-hot vocabularies are fitted on train only; duplicates are removed from
  train only; the test set keeps all its rows.
- *Reproducible.* One command (`python lab04/src/run.py`) rebuilds everything from the raw download. All
  parameters, including the seed (42), live in `config.yaml`; the processed dataset is identified by
  its SHA256, stored with the config in `results/manifest.yaml`.
- *Raw data is never edited.* Validation reports violations and drops nothing.

**Baseline.** Logistic regression on standardized numeric + one-hot categorical features, with and
without two derived features (`capital_net`, `hours_per_age`). The point of the lab is the data
pipeline, so the model stays deliberately simple.

## 2. Ingestion

`ingest.py` downloads the frame and writes it unchanged to `data/raw/adult_raw.csv`: no renaming,
no filling, no type changes. The row and column counts are asserted against the config.

```
ingest: 48842 rows, 15 columns (OpenML id 1590)
```

Raw SHA256: `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc` (`results/ingest.json`).

## 3. Validation

Schema in `config.yaml`: types and ranges from the manual, allowed categories from the UCI data
dictionary (`adult.names`) plus `?`. Rules: `int` columns must be numeric, integral and non-missing;
ranges are age 17–90, hours 1–99, capital-gain 0–99,999, capital-loss 0–4,356; categories must be
in the allowed list; `?` (and NaN) means missing, which is allowed for features and forbidden for
the label.

Result on the raw data: **PASS, 0 violations, 0 rows dropped** (full table in
`results/validation_report.md`). The observed ranges hit the schema limits exactly (age 17–90,
hours 1–99, capital-gain max 99,999, capital-loss max 4,356); all 8 categorical columns match the
dictionary lists with no unknown values.

| Finding | Count | Handling |
|---|---|---|
| Missing `workclass` | 2,799 | allowed; imputed in task 4 |
| Missing `occupation` | 2,809 | allowed; imputed in task 4 |
| Missing `native-country` | 857 | allowed; imputed in task 4 |
| Exact duplicate rows (all 15 columns) | 52 | reported; dropped from train only (task 4) |
| Label missing / unknown | 0 | — |

**Missing values are not `?` in this source.** OpenML version 2 already serves them as empty cells
(NaN); there is not a single `?` in the data. A check written only for `'?'` would therefore report
"0 missing" and pass for the wrong reason. The validator counts both forms separately
(`missing_nan`, `missing_token` columns in the report). The target is called `class` in OpenML; the
config maps it to `income` for validation without editing the raw file.

**Does the validator actually catch anything?** With 0 violations on real data this needs a negative
test. Seven corrupted values injected into 5 rows were all caught:

| Injected | Rule fired |
|---|---|
| `age = 12` | `below_min_17` |
| `age = ?` | `missing_not_allowed` |
| `fnlwgt = abc` | `not_numeric` |
| `capital-loss = 5000` | `above_max_4356` |
| `hours-per-week = 40.5` | `not_integer` |
| `workclass = Pirate` | `unknown_category` |
| `income = ?` | `missing_not_allowed` |

Every violation is written as a row (row id, column, value, rule) to `results/violations.csv`
(empty for the real data); duplicates go to `results/duplicate_rows.csv`.

## 4. Cleaning and transformation

Order in `transform.py`, and why this order:

1. `?` → NaN, `class` → `income`. These are fixed rules from the schema, nothing is fitted, so doing
   them before the split cannot leak.
2. **Split 70/30, stratified, seed 42**, on the raw rows: train 34,189, test 14,653. `>50K` share:
   23.94 % train, 23.93 % test.
3. **Exact duplicates dropped from train only:** 24 rows → train 34,165. Done before fitting, so the
   duplicates do not get extra weight in the medians and percentiles. Test keeps all 14,653 rows: the
   test set should look like the data the model will see, and that data contains duplicates too.
4. **Imputer fitted on train:** median for numeric (no numeric values are missing, so this is a safety
   net), mode for categorical: `workclass → Private`, `occupation → Prof-specialty`,
   `native-country → United-States`.
5. **Clipping to the train 1st / 99th percentiles:**

| Feature | Bounds (train) | Clipped in train (low / high) | Clipped in test (low / high) |
|---|---|---|---|
| age | 17 – 74 | 0 / 302 | 0 / 143 |
| fnlwgt | 27,035 – 509,629 | 342 / 341 | 127 / 145 |
| education-num | 3 – 16 | 240 / 0 | 88 / 0 |
| capital-gain | 0 – 15,024 | 0 / 255 | 0 / 136 |
| capital-loss | 0 – 1,980 | 0 / 340 | 0 / 151 |
| hours-per-week | 8 – 80 | 311 / 226 | 144 / 92 |

6. The fitted `ColumnTransformer` (imputers + clipper) is saved with
   `joblib.dump(pre, "results/preprocessor.joblib")`. Test is only *transformed* with it, never used to
   fit. Loading it needs `lab04/src/` on `sys.path`, because the clipper is a custom class in `common.py`.

**Where the 1st/99th rule hurts.** It works for the long-tailed `fnlwgt`. For other columns it changes
meaning: `capital-gain` is 0 for 92 % of people, so the 99th percentile (15,024) wipes out the
difference between 15K and 99,999, and those are exactly the highest earners. The `hours-per-week`
lower bound turns legal part-time values (1–7 hours, e.g. 4 hours in audit row 3375) into 8. The lab
prescribes the rule, so it is applied as written; a better choice would be to clip only `fnlwgt` and
use `log1p` for the capital columns.

## 5. Derived features

`capital_net = capital-gain − capital-loss` and `hours_per_age = hours-per-week / (age − 16)`,
computed row by row on the cleaned columns, so they carry no statistics of their own. Model:
`StandardScaler` (numeric) + `OneHotEncoder(handle_unknown='ignore')` (categorical) +
`LogisticRegression(max_iter=2000)`, fitted on train and evaluated once on test.

| Features | Accuracy | F1 (>50K) | Inputs after encoding |
|---|---|---|---|
| 14 original | 0.8540 | 0.6601 | 105 |
| + capital_net, hours_per_age | 0.8578 | 0.6696 | 107 |
| Δ | +0.0038 | +0.0095 | |

The gain is small and from a single split (no CI), so it should not be read as more than "does not
hurt, probably helps a little". The likely source is `capital_net`: a linear model on standardized
gain and loss cannot easily express "gain minus loss" on its own. `hours_per_age` has a design flaw:
the denominator is 1 at age 17, so a 17-year-old working 50 hours gets 50 while the train median is
2.0. The feature is driven by the youngest rows.

## 6. Label quality

**Duplicate-conflict audit** (all 48,842 rows, 14 features, NaN treated as equal): 111 rows fall into
54 groups with identical features; in **5 groups (10 rows) the income labels conflict**, i.e. 0.02 %
of the data (`results/label_conflicts.csv`). Typical example: two rows of a 39-year-old married
White male craft-repair worker, Bachelors, 40 h/week, one `<=50K`, one `>50K`. Since even `fnlwgt`
is identical, these are most likely the same Census record labeled twice, or two people the
features cannot tell apart. Either way no model can get both right, so they set a small floor on the
error. They are not exact duplicates (labels differ), so the train dedup keeps both copies.

**50 random rows checked against the data dictionary** (seed 42, `results/label_audit_sample.csv`).
Rule-based checks plus a manual read of the 50 rows:

- Automatic rules: `education` ↔ `education-num` code mapping, `Husband` ⇒ `Male`, `Wife` ⇒ `Female`,
  `Husband`/`Wife` ⇒ married, capital gain and loss both > 0, `Never-worked` with an occupation.
  **0 violations in the sample.**
- Manual read: no impossible values. Five missing cells: two rows (ages 62 and 67) lack both
  `workclass` and `occupation`, consistent with retirement; one lacks `native-country`. Legal but odd:
  `Married-civ-spouse` + `Own-child` (row 23881, age 25), a 40-year-old divorced `Own-child`
  (row 7155), 4 hours/week (row 3375). Labels look plausible against the features (for example
  capital gain 14,344 → `>50K`).

The same rules on the **full** dataset do find problems the sample missed, which shows that 50 rows
cannot detect anything that occurs at a rate below ~1 %:

| Rule | Full dataset |
|---|---|
| education ↔ education-num mismatch | 0 |
| `Wife` with `sex = Male` | 3 |
| `Husband` with `sex = Female` | 1 |
| capital gain and loss both > 0 | 0 |
| Suspicious: capital-gain = 99,999 (top-coded) | 244 |
| Suspicious: hours = 99 / age = 90 (ceiling values) | 137 / 55 |
| Suspicious: under 18 and married | 3 |

The 4 relationship/sex contradictions are rows 5661, 16856, 43422 (`Wife`, `Male`) and 23390
(`Husband`, `Female`). They are reported, not deleted: which of the two fields is wrong cannot be
told from the data. The subjective-label path (relabel after 24 h, Cohen's kappa) does not apply:
Adult labels come from Census records, not from annotators.

## 7. Hash, config, data card, one command

- Processed dataset `data/processed/adult_processed.csv`: train + test rows after the fitted
  preprocessor, plus the 2 derived features and a `split` column, 48,818 rows.
- **SHA256 `adb9e67dbfeb20b544149957e2305010b83541aaffad88af4cb0bc09c2032a2d`**, stored in
  `results/manifest.yaml` together with the full config, the raw-data hash, the preprocessor hash and
  the package versions.
- The CSV is written with a fixed float format and `\n` line endings, so the bytes do not depend on the
  OS or on float printing details.
- **Reproducibility check:** running `run.py` a second time, from a different working directory,
  produced the same SHA256.
- `data_card.md`: source, license, size, known biases, privacy.

One command, from the repo root (paths resolve from the lab folder, so any working directory works),
with the packages from `lab04/requirements.txt`:

```
python lab04/src/run.py
```

## Leakage checklist

| Fitted object | Fitted on |
|---|---|
| Train/test split | raw rows, before any statistic |
| Duplicate removal | train only |
| Median / mode imputer | train only |
| 1st/99th percentile clip bounds | train only |
| StandardScaler, OneHotEncoder, LogisticRegression | train only |
| Derived features | row-wise formula, nothing fitted |
| Metrics | test, once per model |

The one pre-split operation, `?` → NaN, is a fixed rule from the schema and uses no data.
