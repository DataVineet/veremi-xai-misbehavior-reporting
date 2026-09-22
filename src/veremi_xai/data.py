"""Dataset access: one-time CSV -> Parquet conversion and the sender-level split."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ID_COLS, RAW_FEATURES, TARGET, load_config, path


def build_parquet(force: bool = False) -> pd.DataFrame:
    """Read only the useful columns of the 1.2 GB CSV (constant `type` and z-columns are dropped)."""
    out = path("parquet")
    if out.exists() and not force:
        return pd.read_parquet(out)
    cols = ID_COLS + [TARGET] + RAW_FEATURES
    dtypes = {"sender": "int32", "senderPseudo": "int64", "messageID": "int64", TARGET: "int8"}
    df = pd.read_csv(path("raw_csv"), usecols=cols, dtype=dtypes, engine="pyarrow")[cols]
    # The CSV is sender-contiguous and time-sorted inside each sender; make that an explicit guarantee.
    df = df.sort_values(["sender", "sendTime"], kind="stable").reset_index(drop=True)
    df.to_parquet(out, index=False)
    return df


def make_sender_split(df: pd.DataFrame, force: bool = False, seed: int | None = None) -> pd.DataFrame:
    """Class-stratified split at *sender* level (labels are per vehicle -> row splits would leak).

    With `seed=None` the official split file is used / created. A different `seed` returns a fresh split
    in memory only (used by the seed-robustness check) and never touches the official file."""
    out = path("split")
    if seed is None and out.exists() and not force:
        return pd.read_csv(out)
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    fr = cfg["split"]["fractions"]
    senders = df.groupby("sender")[TARGET].first()
    parts = []
    for cls, idx in senders.groupby(senders).groups.items():
        ids = rng.permutation(np.asarray(idx))
        n_tr, n_va = int(round(fr["train"] * len(ids))), int(round(fr["val"] * len(ids)))
        lab = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * (len(ids) - n_tr - n_va))
        parts.append(pd.DataFrame({"sender": ids, TARGET: cls, "split": lab}))
    split = pd.concat(parts, ignore_index=True).sort_values("sender")
    if seed is None:
        split.to_csv(out, index=False)
    return split
