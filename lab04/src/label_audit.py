"""Task 6: label quality. Duplicate-conflict audit + a 50-row check against the data dictionary."""
from common import get_logger, load_config, normalize, read_raw, results_dir, save_json

# adult.names: education-num is the ordinal code of education
EDU_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
           "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
           "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16}
MARRIED = {"Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"}


def dictionary_rules(df):
    """Boolean masks, True = row breaks the rule. 'impossible' contradict the dictionary,
    'suspicious' are legal but worth a look (top-coding, odd combinations)."""
    rel, sex = df["relationship"], df["sex"]
    impossible = {
        "education_num_mismatch": df["education"].map(EDU_NUM) != df["education-num"],
        "husband_not_male": (rel == "Husband") & (sex != "Male"),
        "wife_not_female": (rel == "Wife") & (sex != "Female"),
        "spouse_but_not_married": rel.isin(["Husband", "Wife"]) & ~df["marital-status"].isin(MARRIED),
        "gain_and_loss_both_positive": (df["capital-gain"] > 0) & (df["capital-loss"] > 0),
        "never_worked_with_occupation": (df["workclass"] == "Never-worked") & df["occupation"].notna(),
    }
    suspicious = {
        "capital_gain_topcoded_99999": df["capital-gain"] == 99999,
        "hours_topcoded_99": df["hours-per-week"] == 99,
        "age_topcoded_90": df["age"] == 90,
        "under_18_married": (df["age"] < 18) & df["marital-status"].isin(MARRIED),
    }
    return impossible, suspicious


def main(cfg):
    log = get_logger()
    target = cfg["data"]["target"]
    feats = cfg["numeric"] + cfg["categorical"]
    df = normalize(read_raw(cfg), cfg)
    out = results_dir(cfg)

    # duplicate-conflict: identical 14 features, different income
    groups = df.groupby(feats, dropna=False)[target]
    n_labels = groups.transform("nunique")
    size = groups.transform("size")
    conflict = n_labels > 1
    conflict_rows = df[conflict].sort_values(feats + [target])
    conflict_rows.to_csv(out / "label_conflicts.csv", index_label="row", lineterminator="\n")
    dup_feature_rows = int((size > 1).sum())

    # 50 random rows for the dictionary re-check
    sample = df.sample(n=cfg["label_audit_sample"], random_state=cfg["seed"])
    imp_s, sus_s = dictionary_rules(sample)
    flags = sample.copy()
    flags["flags"] = [", ".join(k for k, m in {**imp_s, **sus_s}.items() if m.loc[i]) for i in sample.index]
    flags.to_csv(out / "label_audit_sample.csv", index_label="row", lineterminator="\n")

    # same rules over the full dataset, to see whether the sample is representative
    imp_all, sus_all = dictionary_rules(df)

    rep = {
        "duplicate_feature_rows": dup_feature_rows,
        "duplicate_feature_groups": int((groups.size() > 1).sum()),
        "conflict_groups": int((groups.nunique() > 1).sum()),
        "conflict_rows": int(conflict.sum()),
        "conflict_rate_of_dataset": round(float(conflict.mean()), 5),
        "sample_size": len(sample),
        "sample_rows": sample.index.tolist(),
        "sample_impossible": {k: int(m.sum()) for k, m in imp_s.items()},
        "sample_suspicious": {k: int(m.sum()) for k, m in sus_s.items()},
        "sample_missing_cells": int(sample[feats].isna().sum().sum()),
        "full_impossible": {k: int(m.sum()) for k, m in imp_all.items()},
        "full_suspicious": {k: int(m.sum()) for k, m in sus_all.items()},
    }
    save_json(rep, out / "label_audit.json")
    log.info("label audit: %d conflict groups (%d rows); sample impossible=%d",
             rep["conflict_groups"], rep["conflict_rows"], sum(rep["sample_impossible"].values()))
    return rep


if __name__ == "__main__":
    main(load_config())
