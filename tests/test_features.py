"""Feature tests: causality (no look-ahead) and row-order preservation.

Needs artifacts/messages.parquet (created by scripts/01_prepare_data.py from the dataset in data/); skipped when it is missing.
Run: python tests/test_features.py
"""
import sys
from pathlib import Path
from unittest import SkipTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from veremi_xai.config import path
from veremi_xai.features import F1, add_features


def _sample():
    if not path("parquet").exists():
        raise SkipTest("artifacts/messages.parquet is missing: place the dataset in data/ and run scripts/01_prepare_data.py")
    df = pd.read_parquet(path("parquet"))
    keep = df.sender.drop_duplicates().sample(40, random_state=1)
    return df[df.sender.isin(keep)].reset_index(drop=True)


def test_features_are_causal_and_order_preserving():
    df = _sample()
    full = add_features(df)
    assert (full.messageID.to_numpy() == df.messageID.to_numpy()).all()          # row order preserved
    # drop the second half of every identity's messages: features of the remaining rows must not change
    first_half = df[df.groupby("senderPseudo").cumcount() < df.groupby("senderPseudo").senderPseudo.transform("size") / 2]
    part = add_features(first_half)
    ref = full.set_index("messageID").loc[part.messageID, F1].to_numpy(dtype=float)
    got = part[F1].to_numpy(dtype=float)
    assert np.allclose(ref, got, equal_nan=True, atol=1e-5), "a feature depends on future messages"


def test_first_message_has_no_history():
    f = add_features(_sample())
    first = f.groupby("senderPseudo").head(1)
    assert first["dt"].isna().all() and (first["msg_index"] == 0).all()


if __name__ == "__main__":
    try:
        test_features_are_causal_and_order_preserving(); test_first_message_has_no_history()
        print("all feature tests passed")
    except SkipTest as e:
        print("SKIPPED:", e)
