"""Step 1b — map-plausibility feature `road_dist` (road map fitted on genuine TRAIN senders only).

Run:  .venv/Scripts/python.exe scripts/01b_road_feature.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from veremi_xai.config import path
from veremi_xai.data import build_parquet, make_sender_split
from veremi_xai.features import RoadMap

df = build_parquet()
split = make_sender_split(df)
lab = df[["sender"]].merge(split[["sender", "split"]], on="sender", how="left")["split"].to_numpy()
rm = RoadMap.fit(df[(lab == "train") & (df["class"] == 0).to_numpy()])
rm.save(path("models") / "road_cells.npy")
rd = pd.DataFrame({"road_dist": rm.distance(df.posx.to_numpy(), df.posy.to_numpy())})
rd.to_parquet(path("features").with_name("road_dist.parquet"), index=False)
print("road cells:", len(rm.cells))
chk = rd.assign(cls=df["class"].to_numpy(), split=lab)
print("share of rows with road_dist > 5 m, TEST split, by class:")
print((chk[chk.split == "test"].assign(off=lambda d: d.road_dist > 5).groupby("cls").off.mean().round(4)).to_dict())
