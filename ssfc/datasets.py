"""Dataset loaders used by the notebooks.

* ``breast_cancer``, ``wine``, ``digits`` ship with scikit-learn (work offline).
* ``pima``, ``ionosphere``, ``sonar``, ``mammography`` are small public CSVs
  downloaded once from a GitHub mirror (jbrownlee/Datasets) into ``data/``.

Every loader returns ``X`` (DataFrame of numeric features), ``y`` (integer
Series) and a short ``info`` string. Constant columns are dropped.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

BASE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

AVAILABLE = ["breast_cancer", "wine", "digits", "pima", "ionosphere", "sonar", "mammography"]


def _download(filename: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / filename
    if not path.exists():
        try:
            urllib.request.urlretrieve(BASE_URL + filename, path)
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"Could not download {filename} ({exc}). Download it manually from "
                f"{BASE_URL}{filename} and place it in {DATA_DIR}/"
            ) from exc
    return path


def _clean(X: pd.DataFrame) -> pd.DataFrame:
    X = X.astype(float)
    return X.loc[:, X.std() > 1e-12]


def load_dataset(name: str):
    """Return ``(X, y, info)`` for one of ``AVAILABLE``."""
    if name == "breast_cancer":
        b = load_breast_cancer()
        X = pd.DataFrame(b.data, columns=[c.replace(" ", "_") for c in b.feature_names])
        y = pd.Series(b.target)
        info = "Breast Cancer Wisconsin (medical, binary)"
    elif name == "wine":
        w = load_wine()
        X = pd.DataFrame(w.data, columns=[c.replace("/", "_") for c in w.feature_names])
        y = pd.Series(w.target)
        info = "Wine chemistry (3 classes)"
    elif name == "digits":
        d = load_digits()
        X = pd.DataFrame(d.data, columns=[f"px{i}" for i in range(d.data.shape[1])])
        y = pd.Series(d.target)
        info = "Handwritten digits 8x8 pixels (10 classes)"
    elif name == "pima":
        df = pd.read_csv(_download("pima-indians-diabetes.csv"), header=None)
        X = df.iloc[:, :-1]
        X.columns = ["pregnancies", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age"]
        y = df.iloc[:, -1].astype(int)
        info = "Pima Indians Diabetes (8 features, binary)"
    elif name == "ionosphere":
        df = pd.read_csv(_download("ionosphere.csv"), header=None)
        X = df.iloc[:, :-1]
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        y = (df.iloc[:, -1] == "g").astype(int)
        info = "Ionosphere radar returns (34 features, binary)"
    elif name == "sonar":
        df = pd.read_csv(_download("sonar.csv"), header=None)
        X = df.iloc[:, :-1]
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        y = (df.iloc[:, -1] == "M").astype(int)
        info = "Sonar mines vs rocks (60 features, binary)"
    elif name == "mammography":
        df = pd.read_csv(_download("mammography.csv"), header=None)
        X = df.iloc[:, :-1]
        X.columns = [f"f{i}" for i in range(X.shape[1])]
        y = (df.iloc[:, -1].astype(str).str.strip("'") == "1").astype(int)
        info = "Mammography (6 features, 11k rows, imbalanced)"
    else:
        raise ValueError(f"unknown dataset '{name}', choose from {AVAILABLE}")
    X = _clean(X).reset_index(drop=True)
    return X, y.reset_index(drop=True).astype(int), info


def load_many(names, verbose=True):
    """Load several datasets, skipping any that cannot be fetched."""
    out = {}
    for n in names:
        try:
            X, y, info = load_dataset(n)
            out[n] = (X, y)
            if verbose:
                print(f"loaded {n:14s} n={X.shape[0]:6d}  d={X.shape[1]:3d}  classes={y.nunique()}  | {info}")
        except Exception as exc:
            print(f"SKIPPED {n}: {exc}")
    return out
