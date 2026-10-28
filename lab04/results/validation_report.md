# Validation report

Rows checked: 48842, rows dropped: 0, status: **PASS**, violations: 0.

Missing columns: none; extra columns: none; renamed: {'class': 'income'}.

| Column | Type | Missing (NaN) | Missing ('?') | Observed | Violations |
|---|---|---|---|---|---|
| age | int | 0 | 0 | 17–90 | 0 |
| workclass | category | 2799 | 0 | 8 categories | 0 |
| fnlwgt | int | 0 | 0 | 12285–1.4904e+06 | 0 |
| education | category | 0 | 0 | 16 categories | 0 |
| education-num | int | 0 | 0 | 1–16 | 0 |
| marital-status | category | 0 | 0 | 7 categories | 0 |
| occupation | category | 2809 | 0 | 14 categories | 0 |
| relationship | category | 0 | 0 | 6 categories | 0 |
| race | category | 0 | 0 | 5 categories | 0 |
| sex | category | 0 | 0 | 2 categories | 0 |
| capital-gain | int | 0 | 0 | 0–99999 | 0 |
| capital-loss | int | 0 | 0 | 0–4356 | 0 |
| hours-per-week | int | 0 | 0 | 1–99 | 0 |
| native-country | category | 857 | 0 | 41 categories | 0 |
| income | category | 0 | 0 | 2 categories | 0 |

Exact duplicate rows (all 15 columns, beyond first occurrence): 52. Reported here, dropped from train only in task 4.
