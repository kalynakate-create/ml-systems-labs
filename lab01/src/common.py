"""Shared helpers for Lab 1: paths, seeds, data split and model definitions."""
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
LAB_DIR = Path(__file__).resolve().parent.parent      # .../repo/lab01
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    """Set every seed required by the lab manual."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch is in requirements.txt, so this only happens if it is not installed


def load_split():
    """Breast Cancer Wisconsin, 70/30 stratified split, random_state=42."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)


def make_models():
    """The two baseline models, exactly as specified in the manual."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }
