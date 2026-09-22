"""Structured evidence package: the ONLY information the reporting layer (template or LLM) receives.

It contains the ML prediction, its confidence, the actual feature values, the top SHAP contributions and
the class definition. It never contains the ground-truth label.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .config import load_config
from .explain import OUTPUT_SPACE
from .features import DESCRIPTIONS

SCHEMA_VERSION = "1.0"
LIMITATIONS = [
    "The detector was trained on simulated traffic (VeReMi Extension, F2MD/LuST); behaviour on real traffic is untested.",
    "Training labels are per vehicle, so some messages of a misbehaving vehicle are behaviourally normal.",
    "SHAP values explain the model's score for this message; they are not proof of a physical cause.",
    "The '*_n' fields are noise/uncertainty values reported with each field; the dataset paper does not define them further.",
]


def _r(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    x = float(x)
    # >= 1: nd decimals; < 1: 4 decimals (never scientific notation such as 1.43e-12, which writers reformat unpredictably)
    return round(x, nd) if abs(x) >= 1 else round(x, 4) + 0.0


def confidence_level(p: float, gap: float) -> str:
    if p >= 0.90 and gap >= 0.50:
        return "high"
    return "moderate" if p >= 0.60 else "low"


def build_evidence(row, proba: np.ndarray, shap_row: np.ndarray, base_values: np.ndarray, features: list[str],
                   model_card: dict, top_k: int = 6) -> dict:
    """row: Series with identifiers + feature values; shap_row: (n_features, n_classes) for this message."""
    classes = load_config()["classes"]
    order = np.argsort(proba)[::-1]
    c = int(order[0])
    sv = shap_row[:, c]
    idx = np.argsort(np.abs(sv))[::-1]
    contrib = []
    for rank, j in enumerate(idx[:top_k], start=1):
        f = features[j]
        contrib.append({"rank": rank, "feature": f, "description": DESCRIPTIONS[f], "value": _r(row[f]),
                        "shap": _r(sv[j]), "effect": "pushes towards the predicted class" if sv[j] > 0 else "pushes away from the predicted class"})
    total = float(np.abs(sv).sum())
    ev = {
        "schema_version": SCHEMA_VERSION,
        "message": {"messageID": int(row["messageID"]), "sendTime_s": _r(row["sendTime"]), "senderPseudo": int(row["senderPseudo"]),
                    "has_history": bool(row.get("msg_index", 0) and row["msg_index"] > 0)},
        "prediction": {"class_id": c, "class_name": classes[c]["name"], "category": classes[c]["group"],
                       "definition": classes[c]["definition"], "confidence": round(float(proba[c]), 4),
                       "confidence_level": confidence_level(float(proba[c]), float(proba[c] - proba[order[1]])),
                       "alternatives": [{"class_id": int(k), "class_name": classes[int(k)]["name"], "probability": round(float(proba[k]), 4)} for k in order[1:3]]},
        "shap": {"explained_class": classes[c]["name"], "output_space": OUTPUT_SPACE, "base_value": _r(base_values[c]),
                 "model_output": _r(base_values[c] + sv.sum()), "top_contributions": contrib,
                 "share_of_total_attribution_in_top": _r(float(np.abs(sv[idx[:top_k]]).sum() / total) if total else 0.0)},
        "transmitted_values": {f: _r(row[f]) for f in ("posx", "posy", "spdx", "spdy", "aclx", "acly", "hedx", "hedy", "spd", "acl")},
        "model": {k: model_card.get(k) for k in ("name", "version", "algorithm", "n_features", "identity", "test_macro_f1", "test_accuracy")},
        "limitations": LIMITATIONS,
    }
    ev["evidence_hash"] = hashlib.sha256(json.dumps(ev, sort_keys=True).encode()).hexdigest()[:16]
    return ev
