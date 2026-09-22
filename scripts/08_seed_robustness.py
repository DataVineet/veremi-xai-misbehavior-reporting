"""Step 8 - does the result depend on which vehicles happened to land in the test set?

The final configuration (F2 features, XGBoost depth 6) is trained again on other class-stratified SENDER splits
(different random seeds). Nothing official is overwritten: splits, features and road maps of these runs live in memory.

Run:  python scripts/08_seed_robustness.py [seed ...]        (default seeds: 1 2 3; about 5 min per seed)
Output: outputs/results/robustness/seed_<n>.json     (aggregated by scripts/09_paper_tables.py)
"""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from veremi_xai import evaluate as ev
from veremi_xai.config import TARGET, class_names, load_config, path
from veremi_xai.data import build_parquet, make_sender_split
from veremi_xai.features import F2, RoadMap, add_features
from veremi_xai.models import make_model

cfg = load_config(); NAMES = class_names()
OUT = path("results") / "robustness"; OUT.mkdir(parents=True, exist_ok=True)
seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
raw = build_parquet()

for seed in seeds:
    f = OUT / f"seed_{seed}.json"
    if f.exists():
        print("exists:", f.name); continue
    t0 = time.time()
    split = make_sender_split(raw, seed=seed)
    df = raw.copy()
    df["split"] = df.sender.map(split.set_index("sender")["split"])
    parts = [add_features(p, identity_col="senderPseudo", window=cfg["features"]["rolling_window"]) for _, p in df.groupby("split", sort=False)]
    feat = pd.concat(parts).sort_index()
    del parts, df; gc.collect()
    tr, va, te = (feat["split"] == s for s in ("train", "val", "test"))
    road = RoadMap.fit(feat[tr & (feat[TARGET] == 0)])             # fitted on genuine TRAIN traffic of this split only
    feat["road_dist"] = road.distance(feat.posx.to_numpy(), feat.posy.to_numpy())
    model = make_model("xgb", cfg["seed"], learning_rate=0.10, max_depth=6, n_estimators=1500)
    model.fit(feat.loc[tr, F2], feat.loc[tr, TARGET], eval_set=[(feat.loc[va, F2], feat.loc[va, TARGET])], verbose=False)
    res = {"seed": seed, "senders": split.groupby("split").size().to_dict(), "best_iteration": int(model.best_iteration), "fit_seconds": round(time.time() - t0, 1)}
    for name, mask in (("val", va), ("test", te)):
        y = feat.loc[mask, TARGET].to_numpy(); pred = model.predict_proba(feat.loc[mask, F2]).argmax(1)
        res[name] = {**ev.summary_metrics(y, pred), **ev.sender_level(y, pred, feat.loc[mask, "sender"])}
        res[name + "_per_class_f1"] = ev.per_class_table(y, pred, NAMES)["f1"].round(4).tolist()
    f.write_text(json.dumps(res, indent=1))
    print(f"seed {seed}: test macro-F1 {res['test']['macro_f1']:.4f}  balanced acc {res['test']['balanced_accuracy']:.4f}  accuracy {res['test']['accuracy']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    del feat, model; gc.collect()
