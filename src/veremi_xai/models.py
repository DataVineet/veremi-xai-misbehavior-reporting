"""Model factory: a deliberately small, justified comparison set for 2.2 M training rows on a laptop."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def xgb_device() -> str:
    try:
        XGBClassifier(n_estimators=1, device="cuda").fit(np.random.rand(50, 2), np.arange(50) % 2)
        return "cuda"
    except Exception:
        return "cpu"


def make_model(name: str, seed: int = 42, **kw):
    if name == "logreg":      # linear baseline; needs imputation + scaling (tree models do not)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=300, C=1.0))
    if name == "tree":        # single interpretable tree
        return DecisionTreeClassifier(max_depth=20, min_samples_leaf=20, random_state=seed)
    if name == "rf":          # bagged trees; max_samples bounds memory on 2.2 M rows
        return RandomForestClassifier(n_estimators=100, max_samples=0.15, min_samples_leaf=5,
                                      n_jobs=-1, random_state=seed)
    if name == "xgb":         # gradient-boosted trees, histogram method; exact TreeSHAP support
        params = dict(n_estimators=600, learning_rate=0.15, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, tree_method="hist", device=xgb_device(), max_bin=256,
                      early_stopping_rounds=30, eval_metric="mlogloss", random_state=seed, n_jobs=-1)
        params.update(kw)
        return XGBClassifier(**params)
    raise ValueError(name)
