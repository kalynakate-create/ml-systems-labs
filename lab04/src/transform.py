"""Task 4: stratified split first, then cleaning fitted on train only.
Order: split -> drop train duplicates -> fit imputer + clipper on train -> transform both."""
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common import (QuantileClipper, get_logger, lab_path, load_config, normalize, read_raw,
                    results_dir, save_json)

INTERIM = "data/interim"


def build_preprocessor(cfg):
    q_low, q_high = cfg["clip_quantiles"]
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper(q_low, q_high))])
    cat = SimpleImputer(strategy="most_frequent")
    ct = ColumnTransformer([("num", num, cfg["numeric"]), ("cat", cat, cfg["categorical"])],
                           verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def main(cfg):
    log = get_logger()
    target = cfg["data"]["target"]
    feats = cfg["numeric"] + cfg["categorical"]
    df = normalize(read_raw(cfg), cfg)

    # split on raw (normalized) rows, before anything is fitted or dropped
    train, test = train_test_split(df, test_size=cfg["split"]["test_size"], random_state=cfg["seed"],
                                   stratify=df[target] if cfg["split"]["stratify"] else None)

    # duplicates: train only; test keeps every row so its metrics stay honest
    n_before = len(train)
    train = train[~train.duplicated(keep="first")]
    dups_dropped = n_before - len(train)

    pre = build_preprocessor(cfg)
    pre.fit(train[feats])
    train_clean = pre.transform(train[feats])
    test_clean = pre.transform(test[feats])
    for clean, part in ((train_clean, train), (test_clean, test)):
        clean[target] = part[target].values
        clean.index = part.index  # original row id, used later by the audit and the hash file

    out = results_dir(cfg)
    joblib.dump(pre, out / "preprocessor.joblib")
    interim = lab_path(INTERIM)
    interim.mkdir(parents=True, exist_ok=True)
    train_clean.to_csv(interim / "train_clean.csv", index_label="row_id", lineterminator="\n")
    test_clean.to_csv(interim / "test_clean.csv", index_label="row_id", lineterminator="\n")

    num_pipe = pre.named_transformers_["num"]
    clipper = num_pipe.named_steps["clip"]
    clipped = {}
    for name, part in (("train", train), ("test", test)):
        filled = num_pipe.named_steps["impute"].transform(part[cfg["numeric"]])
        low = (filled < clipper.lower_).sum(axis=0)
        high = (filled > clipper.upper_).sum(axis=0)
        clipped[name] = {c: {"below": int(l), "above": int(h)}
                         for c, l, h in zip(cfg["numeric"], low, high)}

    pos = cfg["data"]["positive_label"]
    rep = {
        "split": {"train_rows_before_dedup": n_before, "test_rows": len(test),
                  "train_positive_rate": float((train[target] == pos).mean()),
                  "test_positive_rate": float((test[target] == pos).mean())},
        "train_duplicates_dropped": dups_dropped,
        "train_rows_after_dedup": len(train),
        "missing_in_train": {c: int(train[c].isna().sum()) for c in feats if train[c].isna().any()},
        "missing_in_test": {c: int(test[c].isna().sum()) for c in feats if test[c].isna().any()},
        "impute_median": dict(zip(cfg["numeric"],
                                  num_pipe.named_steps["impute"].statistics_.tolist())),
        "impute_mode": dict(zip(cfg["categorical"],
                                pre.named_transformers_["cat"].statistics_.tolist())),
        "clip_bounds": {c: [float(l), float(h)]
                        for c, l, h in zip(cfg["numeric"], clipper.lower_, clipper.upper_)},
        "values_clipped": clipped,
        "fitted_on": "train only",
        "preprocessor": "results/preprocessor.joblib",
    }
    save_json(rep, out / "transform.json")
    log.info("transform: train %d -> %d after dedup (-%d), test %d untouched; preprocessor saved",
             n_before, len(train), dups_dropped, len(test))
    return rep


if __name__ == "__main__":
    main(load_config())
