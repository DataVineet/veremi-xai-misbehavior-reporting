"""End-to-end inference used by BOTH the evaluation scripts and the Streamlit app:

    raw messages of one identity -> features -> ML prediction -> SHAP -> evidence package -> report
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import load_config, path
from .evidence import build_evidence
from .explain import ShapExplainer
from .features import RoadMap, add_features


class Pipeline:
    def __init__(self, model, card: dict, road: RoadMap):
        self.model, self.card, self.road = model, card, road
        self.features, self.identity = card["features"], card["identity"]
        self.explainer = ShapExplainer(model, self.features)
        self.window = load_config()["features"]["rolling_window"]

    @classmethod
    def load(cls) -> "Pipeline":
        d = path("final")
        card = json.loads((d / "model_card.json").read_text(encoding="utf-8"))
        model = XGBClassifier()
        model.load_model(d / card["model_file"])
        return cls(model, card, RoadMap.load(d / "road_cells.npy"))

    def featurize(self, messages: pd.DataFrame) -> pd.DataFrame:
        f = add_features(messages, identity_col=self.identity, window=self.window)
        f["road_dist"] = self.road.distance(f.posx.to_numpy(), f.posy.to_numpy())
        return f

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(feats[self.features])

    def analyse(self, messages: pd.DataFrame, message_id: int, top_k: int = 6) -> dict:
        """`messages`: raw rows available for the identity of `message_id` (earlier rows give the context)."""
        target = messages.loc[messages.messageID == message_id].iloc[0]
        hist = messages[(messages[self.identity] == target[self.identity]) & (messages.sendTime <= target.sendTime)]
        feats = self.featurize(hist)
        row = feats.loc[feats.messageID == message_id]
        proba = self.predict_proba(row)[0]
        shap_row = self.explainer.shap_values(row)[0]
        ev = build_evidence(row.iloc[0], proba, shap_row, self.explainer.base_values, self.features, self.card, top_k)
        return {"evidence": ev, "proba": proba, "shap_row": shap_row, "row": row.iloc[0], "history": feats}
