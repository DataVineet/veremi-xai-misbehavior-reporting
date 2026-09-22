"""Evaluation metrics for the 20-class task (never accuracy alone)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support)

LABELS = list(range(20))


def summary_metrics(y_true, y_pred) -> dict:
    yb_t, yb_p = np.asarray(y_true) > 0, np.asarray(y_pred) > 0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "binary_f1_misbehaving": f1_score(yb_t, yb_p, zero_division=0),   # derived from the same 20-class predictions
    }


def per_class_table(y_true, y_pred, names: dict[int, str]) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return pd.DataFrame({"class": LABELS, "name": [names[c] for c in LABELS], "precision": p, "recall": r, "f1": f, "support": s})


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def sender_level(y_true, y_pred, senders) -> dict:
    """Vehicle-level view: majority vote of a sender's message predictions."""
    d = pd.DataFrame({"s": np.asarray(senders), "t": np.asarray(y_true), "p": np.asarray(y_pred)})
    agg = d.groupby("s").agg(t=("t", "first"), p=("p", lambda x: x.value_counts().index[0]))
    return {"sender_accuracy": accuracy_score(agg.t, agg.p),
            "sender_macro_f1": f1_score(agg.t, agg.p, labels=LABELS, average="macro", zero_division=0)}


def expected_calibration_error(proba: np.ndarray, y_true, bins: int = 15) -> float:
    conf, pred = proba.max(1), proba.argmax(1)
    ok = (pred == np.asarray(y_true)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(ece)
