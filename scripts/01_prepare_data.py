"""Step 1 — CSV -> Parquet, sender-level split, engineered features (both identity modes).

Run:  .venv/Scripts/python.exe scripts/01_prepare_data.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from veremi_xai.config import load_config, path
from veremi_xai.data import build_parquet, make_sender_split
from veremi_xai.features import add_features

t0 = time.time()
cfg = load_config()
df = build_parquet()
split = make_sender_split(df)
print(f"rows={len(df):,}  senders={len(split):,}  [{time.time()-t0:.0f}s]")
print(split.groupby("split").size().to_dict())
df = df.merge(split[["sender", "split"]], on="sender", how="left")
print("rows per split:", df.groupby("split").size().to_dict())

for ident in ("senderPseudo", "sender"):
    out = path("features").with_name(f"features_{ident}.parquet")
    if out.exists():
        print("exists:", out.name); continue
    parts = [add_features(part, identity_col=ident, window=cfg["features"]["rolling_window"]) for _, part in df.groupby("split", sort=False)]
    feat = pd.concat(parts).sort_index()
    feat.to_parquet(out, index=False)
    print(f"{out.name}: {feat.shape}  [{time.time()-t0:.0f}s]")
