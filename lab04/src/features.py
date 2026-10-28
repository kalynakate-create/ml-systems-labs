"""Task 5: derived features, LogisticRegression baseline with and without them."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import add_derived, get_logger, lab_path, load_config, results_dir, save_json
from transform import INTERIM


def load_clean():
    d = lab_path(INTERIM)
    return (pd.read_csv(d / "train_clean.csv", index_col="row_id"),
            pd.read_csv(d / "test_clean.csv", index_col="row_id"))


def fit_eval(train, test, numeric, cfg):
    """Scaler and encoder are fitted inside the model pipeline, i.e. on train only."""
    target, pos = cfg["data"]["target"], cfg["data"]["positive_label"]
    enc = ColumnTransformer([("num", StandardScaler(), numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["categorical"])])
    model = Pipeline([("enc", enc),
                      ("clf", LogisticRegression(max_iter=cfg["logreg"]["max_iter"],
                                                 C=cfg["logreg"]["C"]))])
    cols = numeric + cfg["categorical"]
    model.fit(train[cols], train[target] == pos)
    pred = model.predict(test[cols])
    y = test[target] == pos
    return {"accuracy": round(accuracy_score(y, pred), 4), "f1": round(f1_score(y, pred), 4),
            "n_features_after_encoding": int(model.named_steps["enc"].transform(train[cols][:1]).shape[1]),
            "iterations": int(model.named_steps["clf"].n_iter_[0])}


def main(cfg):
    log = get_logger()
    train, test = load_clean()
    train, test = add_derived(train), add_derived(test)
    base = cfg["numeric"]
    rep = {"without_derived": fit_eval(train, test, base, cfg),
           "with_derived": fit_eval(train, test, base + cfg["derived_features"], cfg)}
    rep["delta"] = {k: round(rep["with_derived"][k] - rep["without_derived"][k], 4)
                    for k in ("accuracy", "f1")}
    rep["derived_stats_train"] = train[cfg["derived_features"]].describe().round(4).to_dict()
    save_json(rep, results_dir(cfg) / "features.json")
    log.info("features: without acc %.4f f1 %.4f | with acc %.4f f1 %.4f",
             rep["without_derived"]["accuracy"], rep["without_derived"]["f1"],
             rep["with_derived"]["accuracy"], rep["with_derived"]["f1"])
    return rep


if __name__ == "__main__":
    main(load_config())
