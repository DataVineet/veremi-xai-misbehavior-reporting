"""Step 10 - showcase examples: clear, instructive TEST messages for the app, the figures and the presentation.

  * saves models/showcase_messages.json  (the app offers these as "Quick examples")
  * draws the local SHAP explanations of all examples (outputs/figures/local_explanations_*.png)
  * optionally writes and caches the reports of a report writer, so that the demo needs neither internet nor quota:
        python scripts/10_showcase.py groq_small

Run:  python scripts/10_showcase.py [backend ...]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from veremi_xai.config import RAW_FEATURES, class_names, load_config, path
from veremi_xai.pipeline import Pipeline
from veremi_xai.reporting import generate_report
from veremi_xai.viz import BLUE, INK2, ORANGE, style

style()
SEED = load_config()["seed"]; NAMES = class_names(); FIG = path("figures")
COLS = ["sendTime", "sender", "senderPseudo", "messageID", "class"] + RAW_FEATURES
demo = pd.read_parquet(path("final") / "demo_messages.parquet").sort_values(["senderPseudo", "sendTime"])
demo["history_len"] = demo.groupby("senderPseudo").cumcount()
picks = demo[demo.in_demo_pick & (demo.senderPseudo != 1)]                 # pseudonym 1 is a placeholder shared by many vehicles
rng = np.random.default_rng(SEED)

WANT = [(13, "correct"), (5, "correct"), (2, "correct"), (3, "correct"), (11, "correct"), (16, "correct"), (18, "correct"), (0, "correct"),
        (9, "wrong"), (2, "wrong"), (12, "wrong"), (16, "wrong"), (17, "wrong"), (10, "wrong")]
MIN_HISTORY = {17: 0, 18: 0, 19: 0}          # these classes change pseudonym all the time, so their identities have almost no history
WRONG_CONFIDENCE = {9: (0.50, 1.00)}         # a still-moving 'eventual stop' vehicle is missed with high confidence
entries = []
for cls, kind in WANT:
    g = picks[(picks["class"] == cls) & (picks.history_len >= MIN_HISTORY.get(cls, 12))]
    lo, hi = WRONG_CONFIDENCE.get(cls, (0.45, 0.92))
    g = g[(g.pred == cls) & (g.conf >= 0.97)] if kind == "correct" else g[(g.pred != cls) & (g.conf.between(lo, hi))]
    if g.empty:
        continue
    m = g.iloc[int(rng.integers(len(g)))]
    label = (f"{cls} {NAMES[cls]}: detected correctly (confidence {m.conf:.2f})" if kind == "correct"
             else f"{cls} {NAMES[cls]}: missed, predicted {NAMES[int(m.pred)]} (confidence {m.conf:.2f})")
    entries.append({"label": label, "outcome": kind, "messageID": int(m.messageID), "sender": int(m.sender), "senderPseudo": int(m.senderPseudo),
                    "true_class": int(cls), "pred": int(m.pred), "conf": round(float(m.conf), 4)})
(path("final") / "showcase_messages.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
print(f"{len(entries)} showcase examples")

pipe = Pipeline.load()
analyses = []
for e in entries:
    hist = demo[demo.senderPseudo == e["senderPseudo"]][COLS]
    analyses.append(pipe.analyse(hist, e["messageID"], top_k=8))


def grid(sel, fname, title):
    n = len(sel); cols_ = 4; rows_ = int(np.ceil(n / cols_))
    fig, axes = plt.subplots(rows_, cols_, figsize=(5.6 * cols_, 3.6 * rows_)); axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, sel):
        e, ev = entries[i], analyses[i]["evidence"]
        c = ev["shap"]["top_contributions"][::-1]
        ax.barh([f"{x['feature']} = {'n/a' if x['value'] is None else x['value']}" for x in c], [x["shap"] for x in c], color=[BLUE if x["shap"] > 0 else ORANGE for x in c], height=0.6)
        ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8)
        ax.set_title(f"true: {NAMES[e['true_class']]}\npredicted: {ev['prediction']['class_name']} ({ev['prediction']['confidence']:.2f})", fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold"); fig.supxlabel("SHAP value for the predicted class (raw margin); blue pushes towards it, orange pushes away", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)


grid([i for i, e in enumerate(entries) if e["outcome"] == "correct"], "local_explanations_correct.png", "Why the model predicted this class: correctly detected messages")
grid([i for i, e in enumerate(entries) if e["outcome"] == "wrong"], "local_explanations_errors.png", "Why the model predicted this class: missed messages")

for backend in sys.argv[1:]:
    for i, (e, a) in enumerate(zip(entries, analyses)):
        ev = pipe.analyse(demo[demo.senderPseudo == e["senderPseudo"]][COLS], e["messageID"], top_k=6)["evidence"]      # top_k 6 = app default -> same cache key
        rep = generate_report(ev, backend=backend)
        print(f"[{backend}] {i+1}/{len(entries)} {e['label'][:48]:48s} written by {rep.backend:10s} valid={rep.validation['passed']} {'(cache)' if rep.from_cache else ''}", flush=True)
        if rep.fallback_reason and "unavailable" in rep.fallback_reason:
            print("  stopped:", rep.fallback_reason[:160]); break
        if not rep.from_cache:
            time.sleep(25)
