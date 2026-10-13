"""
Shared pieces used by both train.py and app.py: data loading, the
chronological split, feature engineering, the PyTorch model definition and
the business-value function. Keeping them here guarantees that the app
transforms data exactly like training did.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
ARTIFACTS = ROOT / "artifacts"

# --- Business assumptions (see REPORT.md, "Business framing") --------------
COST_PER_CALL = 8.0           # EUR: ~6 min of loaded agent time + telephony
VALUE_PER_SUBSCRIPTION = 50.0  # EUR: expected net margin of one term deposit

# --- Split: chronological, the file is ordered May 2008 -> Nov 2010 --------
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remaining 0.15 is the test set

CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "poutcome", "age_band", "month",
]
NUMERIC = ["age", "log_campaign", "previous_capped", "prev_contacted"]
MACRO = ["euribor3m", "cons.price.idx", "cons.conf.idx", "emp.var.rate", "nr.employed"]
# Dropped on purpose (each one is tested as an ablation in train.py):
#   duration     -> only known AFTER the call ends (target leakage)
#   MACRO        -> one value per month for every client, so they cannot rank
#                   clients inside a campaign wave; they only identify the
#                   period, their test values lie outside the training range
#                   and their relation to y flips sign across periods
#   days_since_prev -> filling "never contacted" with 0 made those clients
#                   look like "contacted yesterday"; the flag alone is better
#   day_of_week  -> unstable effect across periods, hurts validation


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    df["y"] = (df["y"] == "yes").astype(int)
    df["t"] = np.arange(len(df))  # row order == contact order
    return df


def chronological_split(df: pd.DataFrame):
    n = len(df)
    i_tr, i_va = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:i_tr].copy(), df.iloc[i_tr:i_va].copy(), df.iloc[i_va:].copy()


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering (no statistics learned from data)."""
    out = df.copy()
    # pdays == 999 is a sentinel for "never contacted before", not a number.
    out["prev_contacted"] = (out["pdays"] != 999).astype(int)
    out["days_since_prev"] = np.where(out["pdays"] == 999, 0, out["pdays"])  # ablation only
    # Number of calls this campaign is heavily right-skewed (max 56).
    out["log_campaign"] = np.log1p(out["campaign"])
    out["previous_capped"] = out["previous"].clip(upper=3)
    # Response vs. age is U-shaped (students and retirees respond more),
    # which a single linear age term cannot represent.
    out["age_band"] = pd.cut(
        out["age"], bins=[0, 25, 35, 50, 60, 200],
        labels=["<=25", "26-35", "36-50", "51-60", "60+"],
    ).astype(str)
    # "default = yes" appears 3 times in 41k rows; fold it into "unknown"
    # (both mean "not a clean no").
    out["default"] = out["default"].replace({"yes": "unknown"})
    return out


class Preprocessor:
    """One-hot + standardisation, fitted on the training split only."""

    def __init__(self, categorical=CATEGORICAL, numeric=NUMERIC):
        self.categorical = list(categorical)
        self.numeric = list(numeric)

    def fit(self, df: pd.DataFrame):
        df = engineer(df)
        self.levels_ = {c: sorted(df[c].unique()) for c in self.categorical}
        self.mean_ = df[self.numeric].mean()
        self.std_ = df[self.numeric].std(ddof=0).replace(0, 1.0)
        self.feature_names_ = self.numeric + [
            f"{c}={lvl}" for c in self.categorical for lvl in self.levels_[c][1:]
        ]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = engineer(df)
        num = ((df[self.numeric] - self.mean_) / self.std_).to_numpy()
        # Drop-first dummies: the first level is the reference category.
        # Unseen levels simply get all-zero dummies (= reference).
        cats = [
            (df[c].to_numpy()[:, None] == np.array(self.levels_[c][1:])[None, :])
            for c in self.categorical
        ]
        return np.hstack([num] + cats).astype(np.float32)


class LogisticRegressionNet(nn.Module):
    """Logistic regression as an nn.Module: one linear layer, sigmoid in the loss."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # logits


def business_value(y_true, proba, threshold):
    """Net campaign profit (EUR) when we call everyone with proba >= threshold."""
    y_true = np.asarray(y_true)
    call = np.asarray(proba) >= threshold
    tp = int(np.sum(call & (y_true == 1)))
    fp = int(np.sum(call & (y_true == 0)))
    fn = int(np.sum(~call & (y_true == 1)))
    tn = int(np.sum(~call & (y_true == 0)))
    profit = tp * (VALUE_PER_SUBSCRIPTION - COST_PER_CALL) - fp * COST_PER_CALL
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "calls": tp + fp, "profit": profit}
