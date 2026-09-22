"""Step 7 - how certain are the results, and why does the model make its mistakes? (final model, TEST set only)

  * bootstrap 95 % confidence intervals (resampling whole VEHICLES within each class, because messages of a vehicle are not independent)
  * paired bootstrap comparison of the feature sets / identity choices
  * calibration, validation of the 'confidence level' used in the reports, selective prediction
  * checks of the explanations given for the main error types (each explanation is tested on the data, not assumed)

Run:  python scripts/07_uncertainty_and_errors.py         (about 4 min)
"""
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
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from veremi_xai.config import class_names, load_config, path
from veremi_xai.evidence import confidence_level
from veremi_xai.features import F0, F1, F2
from veremi_xai.viz import BLUE, CATEGORY_COLORS, INK2, ORANGE, style

style()
RES, FIG = path("results"), path("figures")
NAMES = class_names(); CL = load_config()["classes"]
card = json.loads((path("final") / "model_card.json").read_text())
B = 1000
FEATURE_FILE = lambda ident: path("features").with_name(f"features_{ident}.parquet")
IDS = ["sender", "senderPseudo", "messageID", "class"]

# ---------------------------------------------------------------------------------------------- data
test_mask = pd.read_parquet(FEATURE_FILE("senderPseudo"), columns=["split"])["split"].to_numpy() == "test"
road = pd.read_parquet(path("features").with_name("road_dist.parquet"))["road_dist"].to_numpy()[test_mask]


def test_rows(identity, cols):
    cols = [c for c in cols if c != "road_dist"]
    df = pd.read_parquet(FEATURE_FILE(identity), columns=list(dict.fromkeys(cols + IDS)), filters=[("split", "==", "test")])
    df["road_dist"] = road
    return df.reset_index(drop=True)


def predict(model_file, df, feats):
    m = XGBClassifier(); m.load_model(model_file); m.set_params(device="cpu")
    return m.predict_proba(df[feats])


base = test_rows("senderPseudo", F2)
y, sender = base["class"].to_numpy(), base["sender"].to_numpy()
codes, uniq = pd.factorize(sender)
cls_of = np.zeros(len(uniq), int); cls_of[codes] = y
print(f"test rows {len(base):,} | test vehicles {len(uniq):,}")


def counts(pred):
    c = np.zeros((len(uniq), 20)); np.add.at(c, (codes, pred), 1); return c


proba = predict(path("final") / "final_model.ubj", base, card["features"])
pred, conf = proba.argmax(1), proba.max(1)
srt = np.sort(proba, axis=1); gap = srt[:, -1] - srt[:, -2]
pd.DataFrame({"sender": sender, "messageID": base.messageID, "y": y, "pred": pred, "conf": conf.astype("float32"), "second": np.argsort(proba, axis=1)[:, -2]}
             ).to_parquet(path("parquet").with_name("test_predictions.parquet"), index=False)

VARIANTS = {"F0 (per-message only)": ("xgb_F0", "senderPseudo", F0), "F1 (+history)": ("xgb_F1_pseudo", "senderPseudo", F1),
            "F2 (+road map)": ("xgb_F2_pseudo", "senderPseudo", F2), "F1, history by true sender": ("xgb_F1_sender", "sender", F1),
            "F2, history by true sender": ("xgb_F2_sender", "sender", F2)}
Cs = {"final model": counts(pred)}
for name, (exp, ident, feats) in VARIANTS.items():
    d = base if ident == "senderPseudo" else test_rows(ident, feats)
    assert (d.messageID.to_numpy() == base.messageID.to_numpy()).all()
    Cs[name] = counts(predict(path("models") / f"{exp}.ubj", d, feats).argmax(1))
    stored = json.loads((path("results") / "experiments" / f"{exp}.json").read_text())["test"]["macro_f1"]
    print(f"{name:32s} macro-F1 recomputed vs stored: {stored:.4f}")


# ---------------------------------------------------------------------------------------------- bootstrap
def metrics(cm):
    tp, rows, cols, n = np.diag(cm), cm.sum(1), cm.sum(0), cm.sum()
    rec = np.divide(tp, rows, out=np.zeros(20), where=rows > 0)
    prec = np.divide(tp, cols, out=np.zeros(20), where=cols > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(20), where=(prec + rec) > 0)
    den = np.sqrt((n ** 2 - (cols ** 2).sum()) * (n ** 2 - (rows ** 2).sum()))
    return {"accuracy": tp.sum() / n, "balanced_accuracy": rec.mean(), "macro_f1": f1.mean(), "mcc": (tp.sum() * n - (cols * rows).sum()) / den}, f1, rec, prec


A = {k: c.argmax(1) for k, c in Cs.items()}                       # vehicle-level decision = majority vote of its messages
idx_by_class = [np.where(cls_of == k)[0] for k in range(20)]
rng = np.random.default_rng(load_config()["seed"])
keys = ("accuracy", "balanced_accuracy", "macro_f1", "mcc", "vehicle_macro_f1")
boot = {k: {m: np.zeros(B) for m in keys} for k in Cs}
boot_f1 = np.zeros((B, 20))
for b in range(B):
    sel = [rng.choice(ix, len(ix)) for ix in idx_by_class]
    for k, C in Cs.items():
        m, f1, _, _ = metrics(np.vstack([C[s].sum(0) for s in sel]))
        vcm = np.vstack([np.bincount(A[k][s], minlength=20) for s in sel])
        m["vehicle_macro_f1"] = metrics(vcm)[0]["macro_f1"]
        for kk in keys:
            boot[k][kk][b] = m[kk]
        if k == "final model":
            boot_f1[b] = f1

ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
rows = []
for k, C in Cs.items():
    point, _, _, _ = metrics(np.vstack([C[cls_of == c].sum(0) for c in range(20)]))
    vpoint = metrics(np.vstack([np.bincount(A[k][cls_of == c], minlength=20) for c in range(20)]))[0]["macro_f1"]
    for kk in keys:
        lo, hi = ci(boot[k][kk])
        rows.append({"model": k, "metric": kk, "estimate": point[kk] if kk != "vehicle_macro_f1" else vpoint, "ci_low": lo, "ci_high": hi})
ci_df = pd.DataFrame(rows).round(4); ci_df.to_csv(RES / "test_metrics_ci.csv", index=False)
print(ci_df[ci_df.model == "final model"].to_string(index=False))

pairs = [("F1 (+history)", "F0 (per-message only)"), ("F2 (+road map)", "F1 (+history)"), ("F2 (+road map)", "F2, history by true sender"),
         ("F1 (+history)", "F1, history by true sender")]
pr = []
for a, b_ in pairs:
    d = boot[a]["macro_f1"] - boot[b_]["macro_f1"]; lo, hi = ci(d)
    pr.append({"comparison": f"{a}  minus  {b_}", "mean_macro_f1_difference": d.mean(), "ci_low": lo, "ci_high": hi, "share_of_resamples_positive": (d > 0).mean()})
pd.DataFrame(pr).round(4).to_csv(RES / "paired_comparisons.csv", index=False)
print(pd.DataFrame(pr).round(4).to_string(index=False))

cmf = np.vstack([Cs["final model"][cls_of == c].sum(0) for c in range(20)])
_, f1p, recp, precp = metrics(cmf)
pcls = pd.DataFrame({"class": range(20), "name": [NAMES[c] for c in range(20)], "precision": precp, "recall": recp, "f1": f1p,
                     "f1_ci_low": np.percentile(boot_f1, 2.5, axis=0), "f1_ci_high": np.percentile(boot_f1, 97.5, axis=0),
                     "support_messages": cmf.sum(1).astype(int), "support_vehicles": np.bincount(cls_of, minlength=20)}).round(4)
pcls.to_csv(RES / "final_per_class_metrics_ci.csv", index=False)

# ---------------------------------------------------------------------------------------------- confusions
off = [(int(cmf[i, j]), i, j) for i in range(20) for j in range(20) if i != j]
tot_err = sum(o[0] for o in off)
conf_df = pd.DataFrame([{"true_class": i, "true_name": NAMES[i], "predicted_class": j, "predicted_name": NAMES[j], "messages": n,
                         "share_of_true_class": round(n / cmf[i].sum(), 4), "share_of_all_errors": round(n / tot_err, 4)} for n, i, j in sorted(off, reverse=True)[:15]])
conf_df.to_csv(RES / "top_confusions.csv", index=False)

# ---------------------------------------------------------------------------------------------- calibration, confidence levels, selective prediction
edges = np.linspace(0, 1, 16); ok = (pred == y)
rel = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (conf > lo) & (conf <= hi)
    if m.any():
        rel.append({"bin_low": lo, "bin_high": hi, "n": int(m.sum()), "mean_confidence": conf[m].mean(), "accuracy": ok[m].mean()})
rel = pd.DataFrame(rel); ece = float((rel.n / rel.n.sum() * (rel.accuracy - rel.mean_confidence).abs()).sum())
level = np.where((conf >= 0.90) & (gap >= 0.50), "high", np.where(conf >= 0.60, "moderate", "low"))
for i in np.random.default_rng(1).choice(len(conf), 2000, replace=False):               # the vectorised rule must equal the one used in the evidence package
    assert level[i] == confidence_level(float(conf[i]), float(gap[i]))
lv = pd.DataFrame([{"confidence_level": L, "messages": int((level == L).sum()), "share_of_messages": (level == L).mean(), "accuracy": ok[level == L].mean(),
                    "share_of_all_errors": (~ok & (level == L)).sum() / (~ok).sum()} for L in ("high", "moderate", "low") if (level == L).any()]).round(4)
lv.to_csv(RES / "confidence_level_validation.csv", index=False)
sel = []
for tau in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    k = conf >= tau
    sel.append({"min_confidence": tau, "coverage": k.mean(), "accuracy_kept": ok[k].mean(), "macro_f1_kept": f1_score(y[k], pred[k], labels=list(range(20)), average="macro", zero_division=0),
                "share_of_errors_removed": 1 - (~ok & k).sum() / (~ok).sum()})
sel = pd.DataFrame(sel).round(4); sel.to_csv(RES / "selective_prediction.csv", index=False)
print("ECE (15 bins):", round(ece, 4)); print(lv.to_string(index=False)); print(sel.to_string(index=False))

# ---------------------------------------------------------------------------------------------- error explanations, tested on the data
d = base.copy(); d["pred"] = pred; d["err"] = d.pred != d["class"]
fl = lambda x: round(float(x), 4)
checks = {}
s = d[d["class"] == 9]; mv = s.spd > 0.5
checks["eventual_stop"] = {"claim": "errors are the messages sent BEFORE the vehicle stops", "messages": len(s), "errors": int(s.err.sum()),
                           "share_of_errors_predicted_genuine": fl((s[s.err].pred == 0).mean()), "share_of_errors_with_speed_above_0.5": fl((s[s.err].spd > 0.5).mean()),
                           "error_rate_when_vehicle_is_moving": fl(s[mv].err.mean()), "error_rate_when_vehicle_is_stopped": fl(s[~mv].err.mean())}
s = d[d["class"] == 16]; ph = s.senderPseudo == 1
g = s.groupby(["sender", "senderPseudo"]).err.agg(["sum", "size"]).reset_index()
top = g.sort_values("sum", ascending=False).groupby("sender").head(1)
checks["grid_sybil"] = {"claim": "one identity per attacker (its own, genuine-looking beacons) causes the errors", "messages": len(s), "errors": int(s.err.sum()),
                        "pseudonyms_per_vehicle_median": fl(g.groupby("sender").size().median()),
                        "share_of_errors_in_the_single_worst_pseudonym_of_each_vehicle": fl(top["sum"].sum() / g["sum"].sum()),
                        "error_rate_inside_that_pseudonym": fl(top["sum"].sum() / top["size"].sum()),
                        "share_of_rows_with_placeholder_pseudonym_1": fl(ph.mean()), "error_rate_placeholder_pseudonym_1": fl(s[ph].err.mean()),
                        "error_rate_other_pseudonyms": fl(s[~ph].err.mean())}
# is the worst pseudonym really "the attacker's own genuine-looking identity"?  compare it with the vehicle's other pseudonyms
worst = top.set_index("sender")["senderPseudo"]
s = s.assign(is_worst=s.sender.map(worst) == s.senderPseudo)
s = s[~ph]                                                    # the shared placeholder pseudonym 1 is not a real identity
checks["grid_sybil"].update({"worst_pseudonym_median_distance_to_road_map_m": fl(s[s.is_worst].road_dist.median()), "other_pseudonyms_median_distance_to_road_map_m": fl(s[~s.is_worst].road_dist.median()),
                             "worst_pseudonym_share_of_rows_off_map(>5m)": fl((s[s.is_worst].road_dist > 5).mean()), "other_pseudonyms_share_of_rows_off_map(>5m)": fl((s[~s.is_worst].road_dist > 5).mean()),
                             "worst_pseudonym_median_speed_position_disagreement": fl(s[s.is_worst].r_resid_absmean.median()), "other_pseudonyms_median_speed_position_disagreement": fl(s[~s.is_worst].r_resid_absmean.median())})
s = d[d["class"] == 12]; z = (s.posx == 0) & (s.posy == 0) & (s.spdx == 0) & (s.spdy == 0)
checks["delayed_messages"] = {"claim": "the zero-filled warm-up is easy, the delayed genuine-looking messages are hard", "messages": len(s), "errors": int(s.err.sum()),
                              "share_of_rows_all_zero": fl(z.mean()), "error_rate_zero_rows": fl(s[z].err.mean()), "error_rate_other_rows": fl(s[~z].err.mean()),
                              "share_of_errors_predicted_genuine": fl((s[s.err].pred == 0).mean())}
s = d[d["class"] == 2]; om = s.road_dist > 5
checks["constant_position_offset"] = {"claim": "errors are offsets that still land on a road", "messages": len(s), "errors": int(s.err.sum()), "share_of_rows_off_the_road_map": fl(om.mean()),
                                      "error_rate_off_map": fl(s[om].err.mean()), "error_rate_on_map": fl(s[~om].err.mean())}
first = d.msg_index == 0
checks["first_message_of_an_identity"] = {"claim": "a first message has no history, so it is harder", "share_of_all_test_messages": fl(first.mean()),
                                          "error_rate_first_messages": fl(d[first].err.mean()), "error_rate_other_messages": fl(d[~first].err.mean()),
                                          "share_of_all_errors_that_are_first_messages": fl((d.err & first).sum() / d.err.sum()),
                                          "share_of_messages_that_are_first_by_class": {NAMES[c]: fl(first[d["class"] == c].mean()) for c in (0, 16, 17, 18, 19)}}
s = d[d["class"].isin([17, 19])]
checks["data_replay_sybil_vs_dos_disruptive_sybil"] = {"claim": "the two classes are confused with each other", "class17_predicted_as_19": fl(cmf[17, 19] / cmf[17].sum()),
                                                      "class19_predicted_as_17": fl(cmf[19, 17] / cmf[19].sum()), "class17_recall": fl(recp[17]), "class19_recall": fl(recp[19])}
(RES / "error_cause_checks.json").write_text(json.dumps(checks, indent=1))
print(json.dumps(checks, indent=1))

# ---------------------------------------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(9, 7)); yy = np.arange(20)[::-1]
cols = [CATEGORY_COLORS[CL[c]["group"]] for c in range(20)]
ax.barh(yy, pcls.f1, height=0.6, color=cols, xerr=[pcls.f1 - pcls.f1_ci_low, pcls.f1_ci_high - pcls.f1], error_kw={"ecolor": INK2, "capsize": 2.5, "lw": 1}); ax.grid(axis="y", visible=False)
ax.set_yticks(yy, [f"{n}  {c}" for c, n in zip(pcls["class"], pcls.name)]); ax.set_xlim(0, 1.05); ax.set_xlabel("F1 on the test set, with 95 % bootstrap interval (over vehicles)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()], labels=list(CATEGORY_COLORS), loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
ax.set_title("Per-class F1 with confidence intervals", pad=24); fig.savefig(FIG / "final_per_class_f1_ci.png"); plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
a1.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfect calibration"); a1.plot(rel.mean_confidence, rel.accuracy, color=BLUE, lw=2, marker="o", ms=5, label="final model")
a1.set_xlabel("mean predicted confidence"); a1.set_ylabel("observed accuracy"); a1.set_title(f"Reliability diagram (ECE {ece:.4f})"); a1.legend(loc="upper left")
a2.bar(rel.bin_high - 0.033, rel.n / rel.n.sum(), width=0.06, color=BLUE); a2.set_yscale("log"); a2.set_xlabel("confidence"); a2.set_ylabel("share of test messages (log)"); a2.set_title("How confident is the model?")
fig.tight_layout(); fig.savefig(FIG / "calibration_reliability.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2)); x = np.arange(len(lv)); w = 0.38
ax.bar(x - w / 2, lv.accuracy, w, color=BLUE, label="accuracy inside the level"); ax.bar(x + w / 2, lv.share_of_messages, w, color=ORANGE, label="share of all messages")
for xi, a, s_ in zip(x, lv.accuracy, lv.share_of_messages):
    ax.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=9, color=INK2); ax.text(xi + w / 2, s_ + 0.015, f"{s_:.2f}", ha="center", fontsize=9, color=INK2)
ax.set_xticks(x, lv.confidence_level); ax.set_ylim(0, 1.12); ax.grid(axis="x", visible=False); ax.legend(loc="upper right")
ax.set_title("The confidence level printed in the reports is meaningful"); fig.savefig(FIG / "confidence_levels.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(sel.coverage, sel.macro_f1_kept, color=BLUE, lw=2, marker="o", label="macro-F1 of kept messages"); ax.plot(sel.coverage, sel.accuracy_kept, color=ORANGE, lw=2, marker="s", label="accuracy of kept messages")
for c_, f_, t_ in zip(sel.coverage, sel.macro_f1_kept, sel.min_confidence):
    if t_ in (0.0, 0.8, 0.95, 0.99):
        ax.annotate(f"conf ≥ {t_}", (c_, f_), textcoords="offset points", xytext=(4, -13), fontsize=8, color=INK2)
ax.set_xlabel("coverage (share of messages that are kept)"); ax.set_ylabel("score"); ax.legend(loc="lower left"); ax.set_title("Rejecting low-confidence messages")
fig.savefig(FIG / "selective_prediction.png"); plt.close(fig)

vals = ci_df[ci_df.metric == "macro_f1"].set_index("model"); order = [k for k in vals.index if k != "final model"]
fig, ax = plt.subplots(figsize=(8, 3.8)); yy = np.arange(len(order))[::-1]
ax.errorbar(vals.loc[order, "estimate"], yy, xerr=[vals.loc[order, "estimate"] - vals.loc[order, "ci_low"], vals.loc[order, "ci_high"] - vals.loc[order, "estimate"]], fmt="o", color=BLUE, ecolor=INK2, capsize=4)
ax.set_yticks(yy, order); ax.set_xlabel("macro-F1 on the test set, with 95 % bootstrap interval"); ax.grid(axis="y", visible=False); ax.set_title("Feature sets and identity choice, with uncertainty")
fig.savefig(FIG / "bootstrap_ablation_ci.png"); plt.close(fig)
print("done")
