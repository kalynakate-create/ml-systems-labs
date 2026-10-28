# Data card: UCI Adult, processed for Lab 4

## Source

- **Dataset:** Adult ("Census Income"), extracted by Barry Becker from the 1994 US Census database;
  donated to the UCI Machine Learning Repository by Ronny Kohavi and Barry Becker (1996).
- **Accessed via:** `sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True)`,
  OpenML dataset 1590 (https://www.openml.org/d/1590). Cache: `F:\datasets`.
- **Raw copy:** `data/raw/adult_raw.csv`, SHA256 `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc`.

## License

OpenML metadata: "Public". The UCI repository distributes Adult under Creative Commons Attribution
4.0 (CC BY 4.0): reuse is allowed with attribution to the donors and the repository.

## Size and structure

| | Rows | Columns |
|---|---|---|
| Raw | 48,842 | 15 (14 features + `class`, renamed to `income`) |
| Processed train | 34,165 (34,189 minus 24 exact duplicates) | 17 + `split` |
| Processed test | 14,653 (untouched row set) | 17 + `split` |
| Processed total | 48,818 | `row_id` + 14 features + 2 derived + `income` + `split` |

- Processed file: `data/processed/adult_processed.csv`, SHA256
  `adb9e67dbfeb20b544149957e2305010b83541aaffad88af4cb0bc09c2032a2d` (recorded with the full config in
  `results/manifest.yaml`).
- Fitted preprocessor: `results/preprocessor.joblib` (median/mode imputation, 1st/99th-percentile
  clipping; fitted on train).
- Label: `>50K` is 23.9 % of rows, the same in train and test (stratified split).
- Missing values: `workclass` 2,799, `occupation` 2,809, `native-country` 857. OpenML serves them as
  empty values, not as `?`.

## Known biases and limitations

- **Old and US-only.** Incomes are from 1994 and not adjusted for inflation; 50K then is about
  110K in 2026 dollars. 89.7 % of rows are `native-country = United-States`; most other countries
  have under 100 rows, so per-country patterns are unreliable.
- **Demographic imbalance, with a different base rate per group.** 66.8 % male; 85.5 % White.
  The `>50K` rate is 30.4 % for men and 10.9 % for women, 25.4 % for White and 12.1 % for Black
  people. A model trained on this data reproduces these 1994 gaps; `sex` and `race` are direct inputs,
  and `relationship` (`Husband` / `Wife`) is a proxy for sex.
- **Top-coding.** `capital-gain` = 99,999 in 244 rows and `age` = 90 in 55 rows are ceilings, not
  real values. The pipeline clips `capital-gain` to 15,024 (train 99th percentile), so this signal is
  flattened on purpose (task 4 rule).
- **Label noise.** 5 groups (10 rows) have identical features and opposite labels; 4 rows contradict
  the dictionary on `relationship` vs `sex` (3× `Wife` + `Male`, 1× `Husband` + `Female`). Rows are
  kept and reported in `results/label_audit.json`.
- **Missing not at random.** `workclass` and `occupation` are missing together in all 2,799 rows
  (the other 10 missing `occupation` rows are `Never-worked`). These rows look like people outside the
  labor force: 26.7 % are aged 60+ (7.2 % elsewhere), and only 9.5 % earn `>50K` (23.9 % overall).
  Mode imputation turns them into `Private` / `Prof-specialty`, a poor guess for exactly these people.
- **`fnlwgt`** is the Census sampling weight (how many people the row stands for), not a personal
  attribute. It is kept as a feature only because the lab fixes the input set at 14 features.

## Privacy

- No direct identifiers: no names, addresses, exact dates or IDs. The data was anonymized by the
  Census Bureau before release.
- Re-identification risk is low but not zero: rare combinations (for example a small `native-country`
  group + age + occupation + exact `fnlwgt`) can single out a person in a 1994 Census subsample.
- Sensitive attributes are present: `race`, `sex`, `marital-status`, income. Any model built on this
  data and applied to real people is a discrimination risk; this dataset is for teaching and
  benchmarking only.
