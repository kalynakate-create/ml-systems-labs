"""Task 3: schema, range, missing-value and duplicate checks on the raw data.
Reports every violation; never drops or edits rows."""
import pandas as pd

from common import get_logger, load_config, read_raw, results_dir, save_json


def check(raw, cfg):
    schema = cfg["schema"]
    tokens = set(schema["missing_tokens"])
    df = raw.rename(columns=cfg["data"]["rename"])  # name mapping only, values untouched
    violations = []  # one row per (row, column, rule)
    columns = {}

    expected = list(schema["columns"])
    missing_cols = [c for c in expected if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected]

    for col, rule in schema["columns"].items():
        if col in missing_cols:
            continue
        s = df[col]
        is_nan = s.isna()
        is_token = s.astype(str).str.strip().isin(tokens) & ~is_nan
        missing = is_nan | is_token
        res = {"type": rule["type"], "missing_nan": int(is_nan.sum()),
               "missing_token": int(is_token.sum()), "violations": {}}

        def flag(mask, name):
            res["violations"][name] = int(mask.sum())
            for i in s.index[mask]:
                violations.append({"row": int(i), "column": col, "value": s[i], "rule": name})

        if rule["type"] == "int":
            num = pd.to_numeric(s.where(~missing), errors="coerce")
            flag(~missing & num.isna(), "not_numeric")
            flag(num.notna() & (num != num.round()), "not_integer")
            flag(missing, "missing_not_allowed")  # null policy: '?' only for categories
            if "min" in rule:
                flag(num < rule["min"], f"below_min_{rule['min']}")
            if "max" in rule:
                flag(num > rule["max"], f"above_max_{rule['max']}")
            res.update(observed_min=float(num.min()), observed_max=float(num.max()))
        else:
            allowed = set(rule["allowed"])
            flag(~missing & ~s.astype(str).isin(allowed), "unknown_category")
            if "?" not in allowed:
                flag(missing, "missing_not_allowed")
            res["observed_categories"] = int(s[~missing].nunique())
        columns[col] = res

    dup_mask = df.duplicated(keep="first")
    report = {
        "rows_checked": len(df),
        "rows_dropped": 0,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "renamed": cfg["data"]["rename"],
        "columns": columns,
        "exact_duplicate_rows": int(dup_mask.sum()),
        "total_violations": len(violations),
        "status": "PASS" if not violations and not missing_cols and not extra_cols else "FAIL",
    }
    return report, pd.DataFrame(violations, columns=["row", "column", "value", "rule"]), dup_mask


def to_markdown(rep):
    lines = ["# Validation report", "",
             f"Rows checked: {rep['rows_checked']}, rows dropped: {rep['rows_dropped']}, "
             f"status: **{rep['status']}**, violations: {rep['total_violations']}.", "",
             f"Missing columns: {rep['missing_columns'] or 'none'}; extra columns: "
             f"{rep['extra_columns'] or 'none'}; renamed: {rep['renamed']}.", "",
             "| Column | Type | Missing (NaN) | Missing ('?') | Observed | Violations |",
             "|---|---|---|---|---|---|"]
    for col, r in rep["columns"].items():
        obs = (f"{r['observed_min']:g}–{r['observed_max']:g}" if r["type"] == "int"
               else f"{r['observed_categories']} categories")
        bad = ", ".join(f"{k}: {v}" for k, v in r["violations"].items() if v) or "0"
        lines.append(f"| {col} | {r['type']} | {r['missing_nan']} | {r['missing_token']} | {obs} | {bad} |")
    lines += ["", f"Exact duplicate rows (all 15 columns, beyond first occurrence): "
              f"{rep['exact_duplicate_rows']}. Reported here, dropped from train only in task 4."]
    return "\n".join(lines) + "\n"


def main(cfg):
    log = get_logger()
    raw = read_raw(cfg)
    rep, viol, dup_mask = check(raw, cfg)
    out = results_dir(cfg)
    save_json(rep, out / "validation_report.json")
    (out / "validation_report.md").write_text(to_markdown(rep), encoding="utf-8")
    viol.to_csv(out / "violations.csv", index=False, lineterminator="\n")
    raw[dup_mask].to_csv(out / "duplicate_rows.csv", index_label="row", lineterminator="\n")
    log.info("validate: %s, %d violations, %d exact duplicates, %d rows kept",
             rep["status"], rep["total_violations"], rep["exact_duplicate_rows"], rep["rows_checked"])
    return rep


if __name__ == "__main__":
    main(load_config())
