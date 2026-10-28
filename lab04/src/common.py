"""Lab 4: config, logging, raw-data loading and the custom transformers shared by all steps."""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

LAB_DIR = Path(__file__).resolve().parent.parent  # .../repo/lab04
CONFIG_PATH = LAB_DIR / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def lab_path(rel):
    """Config paths are relative to the lab folder, so the pipeline runs from any cwd."""
    p = Path(rel)
    return p if p.is_absolute() else LAB_DIR / p


def results_dir(cfg):
    d = lab_path(cfg["results_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logger():
    log = logging.getLogger("lab4")
    if not log.handlers:
        (LAB_DIR / "results").mkdir(exist_ok=True)
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        for h in (logging.StreamHandler(),
                  logging.FileHandler(LAB_DIR / "results" / "pipeline.log", mode="a", encoding="utf-8")):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- raw data as the later steps see it ----------

def read_raw(cfg):
    """Raw CSV exactly as ingest.py wrote it: no renaming, no missing-value handling."""
    # keep "?" as a literal string; only empty cells become NaN
    return pd.read_csv(lab_path(cfg["data"]["raw_path"]), keep_default_na=False, na_values=[""])


def normalize(raw, cfg):
    """Stateless rules from the schema (nothing is fitted): rename target, '?' -> NaN.
    Applied before the split, so it cannot leak test information."""
    df = raw.rename(columns=cfg["data"]["rename"])
    df = df.replace(cfg["schema"]["missing_tokens"], np.nan)
    for c in cfg["categorical"] + [cfg["data"]["target"]]:
        df[c] = df[c].astype(object)
    return df


# ---------- Task 4: transformers (live here so joblib.load can find them) ----------

class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [q_low, q_high] percentiles learned in fit (train only)."""

    def __init__(self, q_low=0.01, q_high=0.99):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.q_low, axis=0)
        self.upper_ = np.nanquantile(X, self.q_high, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features, dtype=object)


# ---------- Task 5: derived features ----------

def add_derived(df):
    """Computed row by row from already-cleaned columns, so no statistics are involved."""
    out = df.copy()
    out["capital_net"] = out["capital-gain"] - out["capital-loss"]
    # schema guarantees age >= 17; the floor of 1 only protects against bad input
    out["hours_per_age"] = out["hours-per-week"] / np.maximum(out["age"] - 16, 1)
    return out
