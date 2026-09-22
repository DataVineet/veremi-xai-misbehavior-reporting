"""Step 9 - tables (CSV + Markdown) and the dataset figure for the report / paper.

Everything is computed from the raw Parquet or from saved results. The only typed numbers are the literature rows of
table T11, which are marked as reported by their authors.

Run:  python scripts/09_paper_tables.py
Output: outputs/results/paper_tables/T*.csv / T*.md  and  outputs/figures/class_signatures.png
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from veremi_xai.config import class_names, load_config, path
from veremi_xai.features import CONTEXT, DESCRIPTIONS, F0, KIN
from veremi_xai.viz import CATEGORY_COLORS, INK2, style

style()
RES, FIG = path("results"), path("figures")
OUT = RES / "paper_tables"; OUT.mkdir(exist_ok=True)
CL = load_config()["classes"]; NAMES = class_names()
card = json.loads((path("final") / "model_card.json").read_text())


def save(df, name, note=""):
    df.to_csv(OUT / f"{name}.csv", index=False)
    cols = list(df.columns)
    md = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        md.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    (OUT / f"{name}.md").write_text((note + "\n\n" if note else "") + "\n".join(md) + "\n", encoding="utf-8")
    print(f"{name}: {len(df)} rows")


# ------------------------------------------------------------------------------------------ T0 / T1 / T2 dataset
df = pd.read_parquet(path("parquet"))                                  # sorted by sender, then time
new = df.sender.ne(df.sender.shift())
df["dt"] = df.sendTime.diff().where(~new)
df["spd"] = np.hypot(df.spdx, df.spdy)
h = pd.util.hash_pandas_object(df[KIN], index=False)
genuine_hashes = set(h[df["class"] == 0].to_numpy())
df["dup_any"] = h.duplicated(keep=False)
df["copy_of_genuine"] = h.isin(genuine_hashes) & (df["class"] > 0)
g = df.groupby("class")
pseud = df.groupby(["class", "sender"]).senderPseudo.nunique().groupby("class").median()
t1 = pd.DataFrame({"class_id": range(20), "name": [NAMES[c] for c in range(20)], "group": [CL[c]["group"] for c in range(20)],
                   "messages": g.size().values, "share_of_messages_%": (g.size() / len(df) * 100).round(2).values, "vehicles": g.sender.nunique().values,
                   "median_seconds_between_messages": g.dt.median().round(3).values, "median_pseudonyms_per_vehicle": pseud.values,
                   "mean_speed_m_per_s": g.spd.mean().round(2).values, "share_speed_zero_%": (g.spd.apply(lambda x: (x == 0).mean()) * 100).round(1).values,
                   "share_identical_to_another_message_%": (g.dup_any.mean() * 100).round(1).values,
                   "share_copy_of_a_genuine_message_%": (g.copy_of_genuine.mean() * 100).round(1).values, "definition": [CL[c]["definition"] for c in range(20)]})
save(t1, "T1_dataset_classes", "Table T1. The 20 classes of mixalldata_clean.csv with the measurements that identify them (all computed from the data).")

sha = hashlib.sha256(open(path("raw_csv"), "rb").read()).hexdigest()
t0 = pd.DataFrame([{"item": "file", "value": "data/mixalldata_clean.csv"}, {"item": "sha256", "value": sha}, {"item": "messages", "value": f"{len(df):,}"},
                   {"item": "vehicles (senders)", "value": f"{df.sender.nunique():,}"}, {"item": "pseudonyms", "value": f"{df.senderPseudo.nunique():,}"},
                   {"item": "columns in the csv / used", "value": "30 / 21 (type and 8 z-columns are constant, dropped)"},
                   {"item": "time span (s)", "value": f"{df.sendTime.min():.1f} to {df.sendTime.max():.1f}"},
                   {"item": "genuine messages", "value": f"{(df['class'] == 0).sum():,} ({(df['class'] == 0).mean() * 100:.2f} %)"},
                   {"item": "messages identical (8 kinematic values) to another message", "value": f"{df.dup_any.mean() * 100:.2f} %"},
                   {"item": "attack/fault messages that are copies of a genuine message", "value": f"{df.copy_of_genuine.sum():,}"}])
save(t0, "T0_dataset_overview", "Table T0. Dataset overview.")

split = pd.read_csv(path("split")); df["split"] = df.sender.map(split.set_index("sender")["split"])
t2 = df.groupby(["class", "split"]).agg(vehicles=("sender", "nunique"), messages=("sender", "size")).unstack().fillna(0).astype(int)
t2.columns = [f"{b}_{a}" for a, b in t2.columns]; t2 = t2.reset_index()
t2.insert(1, "name", [NAMES[c] for c in t2["class"]])
save(t2[["class", "name"] + [f"{s}_{k}" for s in ("train", "val", "test") for k in ("vehicles", "messages")]], "T2_split_sizes",
     "Table T2. Class-stratified split by vehicle (70/15/15, seed 42). No vehicle appears in two partitions.")

# ------------------------------------------------------------------------------------------ dataset figure
fig, axes = plt.subplots(1, 4, figsize=(17, 6.2), sharey=True); y = np.arange(20)[::-1]; cols = [CATEGORY_COLORS[CL[c]["group"]] for c in range(20)]
for ax, col, ttl in zip(axes, ("median_seconds_between_messages", "median_pseudonyms_per_vehicle", "mean_speed_m_per_s", "share_copy_of_a_genuine_message_%"),
                        ("Seconds between messages\n(median, one vehicle)", "Pseudonyms per vehicle\n(median)", "Mean transmitted speed\n(m/s)", "Messages that are copies of a\ngenuine message (%)")):
    ax.barh(y, t1[col], height=0.62, color=cols); ax.grid(axis="y", visible=False); ax.set_title(ttl, fontsize=11)
    for yi, v in zip(y, t1[col]):
        ax.text(v, yi, f" {v:g}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, t1[col].max() * 1.18)
axes[0].set_yticks(y, [f"{n}  {c}" for c, n in zip(t1.class_id, t1.name)])
axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()], labels=list(CATEGORY_COLORS), loc="lower right", fontsize=9)
fig.suptitle("What identifies each class in the data (measured on all 3.19 M messages)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "class_signatures.png"); plt.close(fig)

# ------------------------------------------------------------------------------------------ T3 feature dictionary
first_seen = {f: "F0 (per message)" for f in F0}; first_seen.update({f: "F1 (history of the pseudonym)" for f in CONTEXT}); first_seen["road_dist"] = "F2 (road map)"
save(pd.DataFrame([{"feature": f, "first_in_set": first_seen[f], "meaning": DESCRIPTIONS[f]} for f in card["features"]]), "T3_feature_dictionary",
     f"Table T3. The {len(card['features'])} input features of the final detector.")

# ------------------------------------------------------------------------------------------ T4..T7 model results
exps = pd.read_csv(RES / "experiments_summary.csv")
ident = {"senderPseudo": "pseudonym", "sender": "true sender"}
t4 = pd.DataFrame({"experiment": exps.experiment, "model": exps.model, "features": exps.features, "history_keyed_on": exps.identity.map(ident), "split": exps.split.map({"sender": "by vehicle", "row": "random rows"}),
                   "class_weighted": exps.class_weighted, "val_macro_f1": exps.val_macro_f1, "test_macro_f1": exps.test_macro_f1, "test_balanced_accuracy": exps.test_balanced_accuracy,
                   "test_accuracy": exps.test_accuracy, "test_mcc": exps.test_mcc, "train_seconds": exps.fit_seconds}).sort_values("test_macro_f1", ascending=False)
save(t4, "T4_all_experiments", f"Table T4. All {len(t4)} experiments (test set). Selection of the final model used the validation macro-F1 only.")

ci = pd.read_csv(RES / "test_metrics_ci.csv"); pc = pd.read_csv(RES / "paired_comparisons.csv")
mf = ci[ci.metric == "macro_f1"].drop(columns="metric"); mf.columns = ["model", "macro_f1", "ci_low", "ci_high"]
save(mf, "T5a_macro_f1_with_ci", "Table T5a. Test macro-F1 with 95 % bootstrap intervals (1000 resamples of whole vehicles, stratified by class).")
save(pc, "T5b_paired_comparisons", "Table T5b. Paired bootstrap differences of macro-F1 (same resampled vehicles for both models).")

pick = lambda e: exps.set_index("experiment").loc[e, "test_macro_f1"]
t6 = pd.DataFrame([{"features": lab, "split_by_vehicle (correct)": pick(a), "random_row_split (leaky)": pick(b), "inflation": pick(b) - pick(a)}
                   for lab, a, b in (("F0 per message", "xgb_F0", "xgb_F0_rowsplit"), ("F1 + history", "xgb_F1_pseudo", "xgb_F1_pseudo_rowsplit"), ("F2 + road map", "xgb_F2_pseudo", "xgb_F2_pseudo_rowsplit"))])
save(t6, "T6_leakage_effect", "Table T6. Test macro-F1 of the same model and features under the two split strategies.")

fin = ci[ci.model == "final model"].drop(columns="model")
fin["metric"] = fin.metric.map({"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "macro_f1": "macro-F1", "mcc": "MCC", "vehicle_macro_f1": "vehicle-level macro-F1 (majority vote)"})
save(fin, "T7a_final_model_with_ci", f"Table T7a. Final model ({card['experiment']}), test set, 95 % bootstrap intervals.")
rob = sorted((RES / "robustness").glob("seed_*.json")) if (RES / "robustness").exists() else []
if rob:
    rows = [{"split_seed": 42, "test_vehicles": card.get("split", "").split(",")[0] and 3710, **{k: card["test"][k] for k in ("macro_f1", "balanced_accuracy", "accuracy", "mcc")}}]
    for f in rob:
        r = json.loads(f.read_text()); rows.append({"split_seed": r["seed"], "test_vehicles": r["senders"]["test"], **{k: r["test"][k] for k in ("macro_f1", "balanced_accuracy", "accuracy", "mcc")}})
    r7 = pd.DataFrame(rows); mean, sd = r7[["macro_f1", "balanced_accuracy", "accuracy", "mcc"]].mean(), r7[["macro_f1", "balanced_accuracy", "accuracy", "mcc"]].std()
    r7 = pd.concat([r7, pd.DataFrame([{"split_seed": "mean", **mean.to_dict()}, {"split_seed": "std", **sd.to_dict()}])])
    r7["test_vehicles"] = r7["test_vehicles"].map(lambda v: "" if pd.isna(v) else int(v))
    save(r7, "T7b_split_seed_robustness", "Table T7b. Same configuration trained and tested on different vehicle splits (seed 42 = official split).")

t8 = pd.read_csv(RES / "final_per_class_metrics_ci.csv"); t8.insert(2, "group", [CL[c]["group"] for c in t8["class"]])
save(t8, "T8_final_per_class", "Table T8. Per-class test metrics of the final model with bootstrap intervals for F1.")
save(pd.read_csv(RES / "top_confusions.csv"), "T9a_top_confusions", "Table T9a. The 15 most frequent confusions of the final model.")
save(pd.read_csv(RES / "confidence_level_validation.csv"), "T9b_confidence_levels", "Table T9b. Accuracy inside each confidence level (the levels are the ones printed in the reports).")
save(pd.read_csv(RES / "selective_prediction.csv"), "T9c_selective_prediction", "Table T9c. Keeping only messages above a confidence threshold.")

llm = pd.read_csv(RES / "report_faithfulness_summary.csv")
save(llm, "T10_report_faithfulness", "Table T10. Automatic faithfulness validation of generated reports on a class-stratified test sample.")
q = RES / "report_quality_summary.csv"
if q.exists():
    save(pd.read_csv(q), "T10b_report_quality", "Table T10b. Completeness checks of the generated reports (objective rules, no human rating).")
b = RES / "binding_check.json"
if b.exists():
    bind = pd.DataFrame(json.loads(b.read_text())).T.reset_index(drop=True)
    save(bind, "T10c_binding_check", "Table T10c. Offline number-to-feature binding check of the LLM-written reports (lines naming exactly one top feature).")

# ------------------------------------------------------------------------------------------ T11 literature (typed, marked)
lit = pd.DataFrame([
    {"source": "This work", "task": "20 classes, per message", "split": "by vehicle", "features": "per-message only (F0)", "reported": f"macro-F1 {pick('xgb_F0'):.3f}"},
    {"source": "This work", "task": "20 classes, per message", "split": "random rows", "features": "per-message only (F0)", "reported": f"macro-F1 {pick('xgb_F0_rowsplit'):.3f}"},
    {"source": "This work", "task": "20 classes, per message", "split": "by vehicle", "features": "F2: history + road map, no identifiers", "reported": f"macro-F1 {card['test_macro_f1']:.3f}"},
    {"source": "Slama et al. 2022 (same CSV)", "task": "20 classes, per message", "split": "not stated (70/30)", "features": "raw kinematics + noise, identifiers dropped", "reported": "F1 0.696 (random forest)"},
    {"source": "Youness et al. 2025 (VeMisNet)", "task": "20 classes, sequences of 10", "split": "random sequences 80/20", "features": "14 kinematic + communication features", "reported": "accuracy 0.918, F1 0.909, balanced accuracy 0.739"},
    {"source": "Khan et al. 2025", "task": "20 classes, per message", "split": "random rows 49/30/21", "features": "identifiers and noise fields among inputs", "reported": "accuracy 96.15 %"},
    {"source": "Kamel et al. 2020 (dataset paper)", "task": "binary, per message", "split": "not applicable (rule-based)", "features": "plausibility checks", "reported": "F1 0.899 on MixAll"},
])
save(lit, "T11_literature_context", "Table T11. Published numbers on the same data family, as reported by their authors. They are NOT directly comparable (different splits, features, class definitions); see docs/background.md section 6.")
print("done")
