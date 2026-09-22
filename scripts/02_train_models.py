"""Step 2 — experiments: model comparison, feature ablation, leakage demonstration, class weighting.

Run:  .venv/Scripts/python.exe scripts/02_train_models.py [exp ...]     (no args = all missing experiments)
Each experiment writes outputs/results/experiments/<exp>.json; existing results are not recomputed.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from veremi_xai import evaluate as ev
from veremi_xai.config import TARGET, class_names, load_config, path
from veremi_xai.features import F0, F1, F2
from veremi_xai.models import make_model

cfg = load_config(); SEED = cfg["seed"]; NAMES = class_names()
OUT = path("results") / "experiments"; OUT.mkdir(parents=True, exist_ok=True)

# exp name -> (model, feature set, identity used for context, split mode, class-weighted?)
EXPERIMENTS = {
    "xgb_F1_pseudo":          ("xgb",    "F1", "senderPseudo", "sender", False),
    "xgb_F0":                 ("xgb",    "F0", "senderPseudo", "sender", False),
    "xgb_F1_sender":          ("xgb",    "F1", "sender",       "sender", False),
    "xgb_F1_pseudo_weighted": ("xgb",    "F1", "senderPseudo", "sender", True),
    "xgb_F2_pseudo":          ("xgb",    "F2", "senderPseudo", "sender", False),   # + road-map plausibility
    "xgb_F2_sender":          ("xgb",    "F2", "sender",       "sender", False),
    "xgb_F2_pseudo_weighted": ("xgb",    "F2", "senderPseudo", "sender", True),
    "rf_F2_pseudo":           ("rf",     "F2", "senderPseudo", "sender", False),
    "tree_F2_pseudo":         ("tree",   "F2", "senderPseudo", "sender", False),
    "logreg_F2_pseudo":       ("logreg", "F2", "senderPseudo", "sender", False),
    "xgb_F2_pseudo_rowsplit": ("xgb",    "F2", "senderPseudo", "row",    False),   # leakage demonstration
    # small hyper-parameter check around the default (selection on VALIDATION macro-F1 only)
    "xgb_F2_pseudo_lr05":     ("xgb",    "F2", "senderPseudo", "sender", False, dict(learning_rate=0.05, n_estimators=2000, early_stopping_rounds=50)),
    "xgb_F2_pseudo_d10":      ("xgb",    "F2", "senderPseudo", "sender", False, dict(learning_rate=0.10, max_depth=10, n_estimators=1000)),
    "xgb_F2_pseudo_d6":       ("xgb",    "F2", "senderPseudo", "sender", False, dict(learning_rate=0.10, max_depth=6, n_estimators=1500)),
    "rf_F1_pseudo":           ("rf",     "F1", "senderPseudo", "sender", False),
    "tree_F1_pseudo":         ("tree",   "F1", "senderPseudo", "sender", False),
    "logreg_F1_pseudo":       ("logreg", "F1", "senderPseudo", "sender", False),
    "rf_F0":                  ("rf",     "F0", "senderPseudo", "sender", False),
    "xgb_F0_rowsplit":        ("xgb",    "F0", "senderPseudo", "row",    False),   # leakage demonstration
    "xgb_F1_pseudo_rowsplit": ("xgb",    "F1", "senderPseudo", "row",    False),   # leakage demonstration
}
_cache = {}


def load(identity):
    if identity not in _cache:
        _cache.clear()
        _cache[identity] = pd.read_parquet(path("features").with_name(f"features_{identity}.parquet"))
        _cache[identity]["road_dist"] = pd.read_parquet(path("features").with_name("road_dist.parquet"))["road_dist"].to_numpy()
    return _cache[identity]


def run(exp):
    model_name, fset, identity, split_mode, weighted, *extra = EXPERIMENTS[exp]
    kw = extra[0] if extra else {}
    df = load(identity); cols = {"F0": F0, "F1": F1, "F2": F2}[fset]
    if split_mode == "row":   # naive random row split with the same partition sizes (what most papers do)
        rng = np.random.default_rng(SEED); lab = df["split"].to_numpy().copy(); rng.shuffle(lab)
    else:
        lab = df["split"].to_numpy()
    tr, va, te = (lab == s for s in ("train", "val", "test"))
    Xtr, ytr = df.loc[tr, cols], df.loc[tr, TARGET].to_numpy()
    if model_name == "logreg":   # linear baseline on a stratified 400k subsample (full fit is needlessly slow)
        idx = pd.Series(ytr).groupby(ytr).sample(frac=400_000 / len(ytr), random_state=SEED).index
        Xtr, ytr = Xtr.iloc[idx], ytr[idx]
    sw = compute_sample_weight("balanced", ytr) if weighted else None
    model = make_model(model_name, SEED, **kw)
    t0 = time.time()
    if model_name == "xgb":
        model.fit(Xtr, ytr, sample_weight=sw, eval_set=[(df.loc[va, cols], df.loc[va, TARGET])], verbose=False)
    elif model_name == "logreg":
        model.fit(Xtr, ytr)
    else:
        model.fit(Xtr, ytr, sample_weight=sw)
    fit_s = time.time() - t0
    res = {"experiment": exp, "model": model_name, "features": fset, "n_features": len(cols), "identity": identity,
           "split": split_mode, "class_weighted": weighted, "params": kw, "train_rows": int(len(ytr)), "fit_seconds": round(fit_s, 1)}
    if model_name == "xgb":
        res["best_iteration"] = int(model.best_iteration)
    for part, mask in (("val", va), ("test", te)):
        X, y = df.loc[mask, cols], df.loc[mask, TARGET].to_numpy()
        t0 = time.time(); proba = model.predict_proba(X); dt = time.time() - t0
        pred = proba.argmax(1)
        res[part] = {**ev.summary_metrics(y, pred), **ev.sender_level(y, pred, df.loc[mask, "sender"]),
                     "ece": ev.expected_calibration_error(proba, y), "predict_ms_per_1k": round(1e6 * dt / len(y), 2)}
        res[part + "_per_class"] = ev.per_class_table(y, pred, NAMES).round(4).to_dict("records")
        res[part + "_confusion"] = ev.confusion(y, pred).tolist()
    (OUT / f"{exp}.json").write_text(json.dumps(res, indent=1))
    if model_name == "xgb" and split_mode == "sender":
        model.save_model(path("models") / f"{exp}.ubj")
    t = res["test"]
    print(f"{exp:26s} fit {fit_s:6.0f}s | TEST acc {t['accuracy']:.4f} bal-acc {t['balanced_accuracy']:.4f} macro-F1 {t['macro_f1']:.4f} "
          f"MCC {t['mcc']:.4f} bin-F1 {t['binary_f1_misbehaving']:.4f} | sender macro-F1 {t['sender_macro_f1']:.4f}", flush=True)


todo = sys.argv[1:] or [e for e in EXPERIMENTS if not (OUT / f"{e}.json").exists()]
for e in todo:
    run(e)
