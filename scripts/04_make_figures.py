"""Step 4 — result figures from saved experiment JSONs / SHAP files (no training).

Run:  .venv/Scripts/python.exe scripts/04_make_figures.py
Writes NEW files only (names prefixed final_/experiment_/shap_); the original EDA figures are untouched.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from veremi_xai.config import class_names, load_config, path
from veremi_xai.viz import BLUE, CATEGORY_COLORS, INK2, ORANGE, seq_cmap, style

style()
FIG, RES, EXP = path("figures"), path("results"), path("results") / "experiments"
NAMES = class_names(); CL = load_config()["classes"]
card = json.loads((path("final") / "model_card.json").read_text())
final = json.loads((EXP / f"{card['experiment']}.json").read_text())
exps = {p.stem: json.loads(p.read_text()) for p in EXP.glob("*.json")}

# 1. confusion matrix (row-normalised = recall view; magnitude -> one sequential hue)
cm = np.array(final["test_confusion"], dtype=float); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(11, 9)); ax.grid(False)
im = ax.imshow(cmn, cmap=seq_cmap(), vmin=0, vmax=1)
for i in range(20):
    for j in range(20):
        if cmn[i, j] >= 0.005:
            ax.text(j, i, f"{cmn[i, j]:.2f}".lstrip("0") if cmn[i, j] < 1 else "1.0", ha="center", va="center", fontsize=6.5, color="white" if cmn[i, j] > 0.55 else "#0b0b0b")
ax.set_xticks(range(20), [str(i) for i in range(20)]); ax.set_yticks(range(20), [f"{NAMES[i]}  {i}" for i in range(20)])
ax.set_xlabel("predicted class id"); ax.set_ylabel("true class")
ax.set_title(f"Confusion matrix, row-normalised — sender-disjoint test set (macro-F1 {final['test']['macro_f1']:.3f})")
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="share of true-class messages")
fig.savefig(FIG / "final_confusion_matrix.png"); plt.close(fig)

# 2. per-class F1 (identity of category -> fixed categorical colours + legend)
pc = pd.DataFrame(final["test_per_class"]); pc["cat"] = pc["class"].map(lambda c: CL[c]["group"])
fig, ax = plt.subplots(figsize=(9, 7)); y = np.arange(20)[::-1]
ax.barh(y, pc.f1, height=0.62, color=pc.cat.map(CATEGORY_COLORS)); ax.grid(axis="y", visible=False)
for yi, v in zip(y, pc.f1):
    ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, [f"{n}  {c}" for c, n in zip(pc["class"], pc.name)]); ax.set_xlim(0, 1.08); ax.set_xlabel("F1 (test set)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()], labels=list(CATEGORY_COLORS), loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
ax.set_title("Per-class F1 of the final detector", pad=24)
fig.savefig(FIG / "final_per_class_f1.png"); plt.close(fig)

# 3. experiment comparison: ablation / models / leakage (macro-F1, one measure -> one axis each)
def mf(e):
    return exps[e]["test"]["macro_f1"] if e in exps else np.nan
panels = [
    ("Feature sets (XGBoost, sender split)", [("per-message only (F0)", "xgb_F0"), ("+ history, by pseudonym (F1)", "xgb_F1_pseudo"), ("+ road-map plausibility (F2)", "xgb_F2_pseudo"),
                                              ("F1, history by true sender", "xgb_F1_sender"), ("F2, history by true sender", "xgb_F2_sender")]),
    ("Models (F2 features, sender split)", [("Logistic regression", "logreg_F2_pseudo"), ("Decision tree", "tree_F2_pseudo"), ("Random forest", "rf_F2_pseudo"),
                                            ("XGBoost", "xgb_F2_pseudo"), ("XGBoost, class-weighted", "xgb_F2_pseudo_weighted")]),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1]})
for ax, (title, items) in zip(axes[:2], panels):
    items = [(l, e) for l, e in items if e in exps]; v = [mf(e) for _, e in items]; yy = np.arange(len(items))[::-1]
    ax.barh(yy, v, height=0.55, color=BLUE); ax.grid(axis="y", visible=False)
    for yi, x in zip(yy, v):
        ax.text(x + 0.01, yi, f"{x:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(yy, [l for l, _ in items]); ax.set_xlim(0, 1.12); ax.set_xlabel("macro-F1 (test)"); ax.set_title(title)
ax = axes[2]; pairs = [("F0", "xgb_F0", "xgb_F0_rowsplit"), ("F1", "xgb_F1_pseudo", "xgb_F1_pseudo_rowsplit"), ("F2", "xgb_F2_pseudo", "xgb_F2_pseudo_rowsplit")]
pairs = [p for p in pairs if p[1] in exps and p[2] in exps]; x = np.arange(len(pairs)); w = 0.36
for off, idx, col, lab in ((-w / 2 - 0.01, 1, BLUE, "sender-disjoint split (ours)"), (w / 2 + 0.01, 2, ORANGE, "random row split (leaky)")):
    vals = [mf(p[idx]) for p in pairs]; ax.bar(x + off, vals, w, color=col, label=lab)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=INK2)
ax.set_xticks(x, [p[0] for p in pairs]); ax.set_ylim(0, 1.15); ax.grid(axis="x", visible=False); ax.set_ylabel("macro-F1 (test)")
ax.set_title("Leakage effect of the split", pad=34); ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "experiment_comparison.png"); plt.close(fig)

# 4. SHAP: global bar + per-class heatmap
g = pd.read_csv(RES / "shap_global_importance.csv", index_col=0).iloc[:20, 0][::-1]
fig, ax = plt.subplots(figsize=(8, 6.5)); ax.barh(g.index, g.values, height=0.6, color=BLUE); ax.grid(axis="y", visible=False)
ax.set_xlabel("mean |SHAP| over all 20 class scores (raw margin)"); ax.set_title("Global feature importance (exact TreeSHAP, 3,000 test messages)")
fig.savefig(FIG / "shap_global_importance.png"); plt.close(fig)

pcs = pd.read_csv(RES / "shap_per_class_importance.csv", index_col=0)
top = pcs.max(axis=1).sort_values(ascending=False).index[:22]; H = pcs.loc[top]; Hn = H / H.max(axis=0)
fig, ax = plt.subplots(figsize=(12, 8)); ax.grid(False)
im = ax.imshow(Hn.to_numpy(), cmap=seq_cmap(), vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(20), [f"{i} {n}" for i, n in enumerate(H.columns)], rotation=55, ha="right"); ax.set_yticks(range(len(top)), top)
ax.set_title("Which features drive each class — mean |SHAP| of the class's own score, scaled per class (column max = 1)")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="relative importance within class")
fig.savefig(FIG / "shap_per_class_heatmap.png"); plt.close(fig)
print("figures written to", FIG)

# 5. tables: all experiments + final per-class (CSV, for the report/README)
rows = [{"experiment": k, "model": r["model"], "features": r["features"], "n_features": r["n_features"], "identity": r["identity"], "split": r["split"],
         "class_weighted": r["class_weighted"], "val_macro_f1": r["val"]["macro_f1"], **{f"test_{m}": r["test"][m] for m in
         ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1", "mcc", "binary_f1_misbehaving", "sender_macro_f1", "ece", "predict_ms_per_1k")},
         "fit_seconds": r["fit_seconds"]} for k, r in exps.items()]
pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False).round(4).to_csv(RES / "experiments_summary.csv", index=False)
pc.drop(columns="cat").round(4).to_csv(RES / "final_per_class_metrics.csv", index=False)
print(pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False).round(4)[["experiment", "val_macro_f1", "test_macro_f1", "test_balanced_accuracy", "test_accuracy"]].to_string(index=False))
