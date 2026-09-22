"""Feature engineering.

F0  = per-message features (16 raw fields + speed/acceleration magnitude).
F1  = F0 + *causal* context features computed from the current and earlier messages of the same
      identity (no look-ahead, so a message can be scored the moment it arrives).

The same function is used for training, evaluation and the Streamlit app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_FEATURES

KIN = ["posx", "posy", "spdx", "spdy", "aclx", "acly", "hedx", "hedy"]
F0 = RAW_FEATURES + ["spd", "acl"]
CONTEXT = [
    "dt", "dpos", "v_implied", "spd_resid", "vel_err_x", "vel_err_y", "acl_err_x", "acl_err_y",
    "hed_disp_cos", "dhed", "same_as_prev", "msg_index",
    "r_dt_mean", "r_dpos_mean", "r_spd_std", "r_resid_absmean", "r_resid_std",
    "r_vel_err_x_mean", "r_vel_err_y_mean", "r_vel_err_x_std", "r_vel_err_y_std", "r_same_frac",
]
F1 = F0 + CONTEXT

DESCRIPTIONS = {
    "posx": "transmitted X position (m)", "posy": "transmitted Y position (m)",
    "posx_n": "noise/uncertainty value reported with X position", "posy_n": "noise/uncertainty value reported with Y position",
    "spdx": "transmitted X speed (m/s)", "spdy": "transmitted Y speed (m/s)",
    "spdx_n": "noise/uncertainty value reported with X speed", "spdy_n": "noise/uncertainty value reported with Y speed",
    "aclx": "transmitted X acceleration (m/s^2)", "acly": "transmitted Y acceleration (m/s^2)",
    "aclx_n": "noise/uncertainty value reported with X acceleration", "acly_n": "noise/uncertainty value reported with Y acceleration",
    "hedx": "transmitted heading, X component", "hedy": "transmitted heading, Y component",
    "hedx_n": "noise/uncertainty value reported with heading X", "hedy_n": "noise/uncertainty value reported with heading Y",
    "spd": "transmitted speed magnitude (m/s)", "acl": "transmitted acceleration magnitude (m/s^2)",
    "dt": "time since the previous message of the same identity (s); genuine beacons arrive every 1 s",
    "dpos": "distance between this and the previous transmitted position (m)",
    "v_implied": "speed implied by the change in transmitted position, dpos/dt (m/s)",
    "spd_resid": "transmitted speed minus position-implied speed (m/s); near 0 when position and speed agree",
    "vel_err_x": "transmitted X speed minus position-implied X velocity (m/s)",
    "vel_err_y": "transmitted Y speed minus position-implied Y velocity (m/s)",
    "acl_err_x": "transmitted X acceleration minus speed-implied X acceleration (m/s^2)",
    "acl_err_y": "transmitted Y acceleration minus speed-implied Y acceleration (m/s^2)",
    "hed_disp_cos": "cosine between transmitted heading and direction of travel implied by positions (1 = aligned)",
    "dhed": "change of heading since the previous message (0 = unchanged, 2 = reversed)",
    "same_as_prev": "1 if position, speed, acceleration and heading are all identical to the previous message",
    "msg_index": "number of earlier messages seen from this identity (capped at 50)",
    "r_dt_mean": "mean time between the last messages of this identity (s)",
    "r_dpos_mean": "mean position jump over the last messages (m)",
    "r_spd_std": "standard deviation of transmitted speed over the last messages (m/s)",
    "r_resid_absmean": "mean absolute speed-vs-position disagreement over the last messages (m/s)",
    "r_resid_std": "standard deviation of the speed-vs-position disagreement over the last messages (m/s)",
    "r_vel_err_x_mean": "mean X velocity error over the last messages (m/s); a stable non-zero value indicates a constant bias",
    "r_vel_err_y_mean": "mean Y velocity error over the last messages (m/s); a stable non-zero value indicates a constant bias",
    "r_vel_err_x_std": "standard deviation of the X velocity error over the last messages (m/s)",
    "r_vel_err_y_std": "standard deviation of the Y velocity error over the last messages (m/s)",
    "r_same_frac": "fraction of the last messages that exactly repeated their predecessor",
}


def add_features(df: pd.DataFrame, identity_col: str = "senderPseudo", window: int = 10) -> pd.DataFrame:
    """Return a copy of `df` with F1 columns added. Row order of the input is preserved.

    Context is taken only from earlier rows with the same `identity_col` value *inside `df`*;
    call it separately per data partition so no test feature can depend on training rows.
    """
    out = df.copy()
    out["_row"] = np.arange(len(out))
    out = out.sort_values([identity_col, "sendTime"], kind="stable")
    g = out.groupby(identity_col, sort=False)
    first = g.cumcount().to_numpy() == 0

    def prev(col):
        return g[col].shift()

    out["spd"] = np.hypot(out.spdx, out.spdy)
    out["acl"] = np.hypot(out.aclx, out.acly)
    dt = out.sendTime - prev("sendTime")
    dx, dy = out.posx - prev("posx"), out.posy - prev("posy")
    out["dt"] = dt
    out["dpos"] = np.hypot(dx, dy)
    safe_dt = dt.where(dt > 1e-6)
    out["v_implied"] = out.dpos / safe_dt
    out["spd_resid"] = out.spd - out.v_implied
    out["vel_err_x"] = out.spdx - dx / safe_dt
    out["vel_err_y"] = out.spdy - dy / safe_dt
    out["acl_err_x"] = out.aclx - (out.spdx - prev("spdx")) / safe_dt
    out["acl_err_y"] = out.acly - (out.spdy - prev("spdy")) / safe_dt
    moved = out.dpos > 0.5
    hn = np.hypot(out.hedx, out.hedy)
    out["hed_disp_cos"] = ((out.hedx * dx + out.hedy * dy) / (hn * out.dpos)).where(moved & (hn > 0))
    out["dhed"] = 1.0 - (out.hedx * prev("hedx") + out.hedy * prev("hedy"))
    same = np.ones(len(out), dtype=bool)
    for c in KIN:
        same &= (out[c] == prev(c)).to_numpy()
    out["same_as_prev"] = np.where(first, np.nan, same.astype(float))
    out["msg_index"] = np.minimum(g.cumcount(), 50).astype("float32")

    out["_absres"] = out.spd_resid.abs()
    roll = {"r_dt_mean": ("dt", "mean"), "r_dpos_mean": ("dpos", "mean"), "r_spd_std": ("spd", "std"),
            "r_resid_absmean": ("_absres", "mean"), "r_resid_std": ("spd_resid", "std"),
            "r_vel_err_x_mean": ("vel_err_x", "mean"), "r_vel_err_y_mean": ("vel_err_y", "mean"),
            "r_vel_err_x_std": ("vel_err_x", "std"), "r_vel_err_y_std": ("vel_err_y", "std"),
            "r_same_frac": ("same_as_prev", "mean")}
    cols = sorted({c for c, _ in roll.values()})
    r = out.groupby(identity_col, sort=False)[cols].rolling(window, min_periods=2)
    means, stds = r.mean().reset_index(level=0, drop=True), r.std().reset_index(level=0, drop=True)
    for name, (col, how) in roll.items():
        out[name] = (means if how == "mean" else stds)[col]

    out = out.sort_values("_row").drop(columns=["_row", "_absres"])
    ctx = [c for c in F1 if c not in RAW_FEATURES]
    out[ctx] = out[ctx].replace([np.inf, -np.inf], np.nan).astype("float32")
    return out


# ----------------------------------------------------------------------------------------------
# F2 = F1 + map plausibility. The "road map" is learned from GENUINE TRAINING vehicles only
# (a fitted preprocessing step, like a scaler) and stored as an artifact for inference.
# ----------------------------------------------------------------------------------------------
F2 = F1 + ["road_dist"]
DESCRIPTIONS["road_dist"] = ("distance from the transmitted position to the nearest location ever occupied by genuine "
                             "training traffic (m); genuine vehicles are almost always within a few metres")


class RoadMap:
    """Set of 5 m grid cells visited by genuine training vehicles + nearest-cell distance query."""

    def __init__(self, cells: np.ndarray, cell_m: float = 5.0):
        from scipy.spatial import cKDTree
        self.cells, self.cell_m = cells, cell_m
        self._tree = cKDTree((cells + 0.5) * cell_m)

    @classmethod
    def fit(cls, genuine_train: pd.DataFrame, cell_m: float = 5.0) -> "RoadMap":
        xy = np.floor(genuine_train[["posx", "posy"]].to_numpy() / cell_m).astype(np.int32)
        return cls(np.unique(xy, axis=0), cell_m)

    def distance(self, posx, posy, cap: float = 200.0) -> np.ndarray:
        d, _ = self._tree.query(np.column_stack([posx, posy]), workers=-1)
        return np.minimum(d, cap).astype("float32")

    def save(self, file) -> None:
        np.save(file, self.cells)

    @classmethod
    def load(cls, file) -> "RoadMap":
        return cls(np.load(file))
