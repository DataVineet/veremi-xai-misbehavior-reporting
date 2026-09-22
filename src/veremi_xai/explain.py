"""SHAP explainability for the trained tree model (exact TreeSHAP via the `shap` library).

Output space: SHAP values are in the model's *raw margin* (log-odds-like score before softmax) of one
class. They are additive: base_value[c] + sum(shap[:, c]) == margin[c]. They are NOT probability points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

OUTPUT_SPACE = "raw margin (log-odds score before softmax) of the explained class"


class ShapExplainer:
    def __init__(self, model, feature_names: list[str]):
        model.set_params(device="cpu")          # TreeSHAP runs on CPU; keeps the app GPU-independent
        self.model, self.features = model, list(feature_names)
        self._ex = shap.TreeExplainer(model)
        # For XGBoost, shap only sets the correct expected_value during the first shap_values() call,
        # so trigger one on a dummy row before reading the base values (otherwise additivity breaks).
        self._ex.shap_values(pd.DataFrame(np.zeros((1, len(self.features)), dtype="float32"), columns=self.features))
        self.base_values = np.asarray(self._ex.expected_value, dtype=float)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """-> array (n_rows, n_features, n_classes)."""
        sv = np.asarray(self._ex.shap_values(X[self.features]))
        self.base_values = np.asarray(self._ex.expected_value, dtype=float)
        return sv if sv.shape[1] == len(self.features) else np.moveaxis(sv, 0, -1)

    def margins(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features], output_margin=True)

    def additivity_error(self, X: pd.DataFrame) -> float:
        sv = self.shap_values(X)
        return float(np.abs(sv.sum(1) + self.base_values - self.margins(X)).max())
