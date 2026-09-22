"""Step 3 — freeze the final model, build app artifacts, compute SHAP (global + sanity checks).

Run:  .venv/Scripts/python.exe scripts/03_finalize_and_explain.py [experiment_name]
The final experiment defaults to the best VALIDATION macro-F1 among sender-split XGBoost runs that use the
receiver-observable identity (selection never looks at the test set).
"""
import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from veremi_xai.config import ID_COLS, RAW_FEATURES, TARGET, class_names, load_config, path
from veremi_xai.features import F0, F1, F2
from veremi_xai.pipeline import Pipeline

cfg = load_config(); SEED = cfg["seed"]; rng = np.random.default_rng(SEED)
EXP = path("results") / "experiments"; FINAL = path("final"); FINAL.mkdir(exist_ok=True)
RES = path("results")

# ---- 1. select + freeze -------------------------------------------------------------------------
cands = [json.loads(p.read_text()) for p in EXP.glob("xgb_*.json")]
cands = [c for c in cands if c["split"] == "sender" and c["identity"] == "senderPseudo"]
name = sys.argv[1] if len(sys.argv) > 1 else max(cands, key=lambda c: c["val"]["macro_f1"])["experiment"]
r = json.loads((EXP / f"{name}.json").read_text())
feats = {"F0": F0, "F1": F1, "F2": F2}[r["features"]]
# Keep only the trees up to the early-stopping optimum. Predictions are unchanged, and TreeSHAP becomes exactly
# additive w.r.t. the deployed model (with the extra 30 early-stopping rounds left in, shap disagreed with predict()).
from xgboost import XGBClassifier
_m = XGBClassifier(); _m.load_model(path("models") / f"{name}.ubj")
_m.get_booster()[: r["best_iteration"] + 1].save_model(FINAL / "final_model.ubj")
shutil.copy(path("models") / "road_cells.npy", FINAL / "road_cells.npy")
card = {"name": "veremi-xai-detector", "version": "1.0", "date": str(date.today()), "experiment": name,
        "algorithm": "XGBoost multiclass (hist), softprob, early stopping on validation mlogloss",
        "model_file": "final_model.ubj", "feature_set": r["features"], "features": feats, "n_features": len(feats),
        "identity": r["identity"], "class_weighted": r["class_weighted"], "best_iteration": r.get("best_iteration"),
        "split": "sender-disjoint, class-stratified 70/15/15 (artifacts/sender_split.csv)", "train_rows": r["train_rows"],
        "val": r["val"], "test": r["test"], "test_macro_f1": round(r["test"]["macro_f1"], 4), "test_accuracy": round(r["test"]["accuracy"], 4)}
(FINAL / "model_card.json").write_text(json.dumps(card, indent=1))
print("final model:", name, "| val macro-F1", round(r["val"]["macro_f1"], 4), "| test macro-F1", card["test_macro_f1"])

# ---- 2. demo data for the app: whole TEST senders only ---------------------------------------------
raw = pd.read_parquet(path("parquet"))
split = pd.read_csv(path("split"))
test_s = split[split.split == "test"]
pick = test_s.groupby(TARGET, group_keys=False).sample(n=25, random_state=SEED).sender
demo = raw[raw.sender.isin(pick) | (raw.sender.isin(test_s.sender) & (raw.senderPseudo == 1))].copy()
pipe = Pipeline.load()
df = pipe.featurize(demo)
proba = pipe.predict_proba(df)
demo["pred"], demo["conf"] = proba.argmax(1).astype("int8"), proba.max(1).astype("float32")
demo["in_demo_pick"] = demo.sender.isin(pick)
demo[ID_COLS + [TARGET] + RAW_FEATURES + ["pred", "conf", "in_demo_pick"]].to_parquet(FINAL / "demo_messages.parquet", index=False)
print("demo rows:", len(demo), "| demo accuracy:", round(float((demo.pred == demo[TARGET]).mean()), 4))

# ---- 3. SHAP on a class-stratified sample of TEST messages ---------------------------------------
full = pd.read_parquet(path("features").with_name(f"features_{r['identity']}.parquet"))
full["road_dist"] = pd.read_parquet(path("features").with_name("road_dist.parquet"))["road_dist"].to_numpy()
test = full[full.split == "test"]
samp = test.groupby(TARGET, group_keys=False).sample(n=150, random_state=SEED)
sv = pipe.explainer.shap_values(samp)                      # (n, features, classes)
proba_s = pipe.predict_proba(samp); pred_s = proba_s.argmax(1)
add_err = float(np.abs(sv.sum(1) + pipe.explainer.base_values - pipe.explainer.margins(samp)).max())
np.savez_compressed(path("models") / "shap_sample.npz", shap=sv.astype("float32"), X=samp[feats].to_numpy("float32"),
                    y=samp[TARGET].to_numpy(), pred=pred_s, features=np.array(feats))
NAMES = class_names()
overall = pd.Series(np.abs(sv).mean(axis=(0, 2)), index=feats).sort_values(ascending=False)
overall.rename("mean_abs_shap").to_csv(RES / "shap_global_importance.csv")
y = samp[TARGET].to_numpy()
per_class = pd.DataFrame({NAMES[c]: np.abs(sv[y == c][:, :, c]).mean(0) for c in range(20)}, index=feats)
per_class.to_csv(RES / "shap_per_class_importance.csv")

# ---- 4. faithfulness sanity check: replace top-3 SHAP features vs 3 random features ----------------
bg = test[test[TARGET] == 0].sample(len(samp), random_state=SEED)[feats].to_numpy()
X = samp[feats].to_numpy().copy(); n = len(X); k = 3
p0 = proba_s[np.arange(n), pred_s]
top = np.argsort(-np.abs(sv[np.arange(n), :, pred_s]), axis=1)[:, :k]
rnd = np.array([rng.choice(len(feats), k, replace=False) for _ in range(n)])
drops = {}
for label, idx in (("top3_shap", top), ("random3", rnd)):
    Xp = X.copy()
    for j in range(k):
        Xp[np.arange(n), idx[:, j]] = bg[np.arange(n), idx[:, j]]
    p1 = pipe.model.predict_proba(pd.DataFrame(Xp, columns=feats))[np.arange(n), pred_s]
    drops[label] = {"mean_prob_drop": float((p0 - p1).mean()), "share_prediction_flipped": float((pipe.model.predict_proba(pd.DataFrame(Xp, columns=feats)).argmax(1) != pred_s).mean())}
mis = pred_s != 0                                            # perturbation is only meaningful for non-genuine predictions
sanity = {"n_explained": int(n), "additivity_max_abs_error": add_err, "perturbation_all": drops,
          "note": "features replaced by values of a random genuine test message; probability of the originally predicted class"}
(RES / "shap_sanity_checks.json").write_text(json.dumps(sanity, indent=1))
print("SHAP additivity max error:", add_err); print(json.dumps(drops, indent=1))
print("top global features:", overall.head(10).round(3).to_dict())
