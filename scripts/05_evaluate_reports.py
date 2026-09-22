"""Step 5 - report generation + automatic faithfulness evaluation on a stratified test sample.

Run:  python scripts/05_evaluate_reports.py [backend] [n_per_class] [--patient]
      backend: template (default) | groq | groq_small | gemini | openrouter | ollama      n_per_class: default 10
      --patient: when a free-tier limit is hit, wait for the time the provider announces (max 4 h in total) and go on
Only this small sample is ever sent to a report writer, never the dataset. Finished reports are cached, so re-runs are free.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # Windows consoles use cp1252; LLM text can contain any Unicode

from veremi_xai.config import RAW_FEATURES, load_config, path
from veremi_xai.pipeline import Pipeline
from veremi_xai.reporting import generate_report

args = [a for a in sys.argv[1:] if not a.startswith("--")]
patient = "--patient" in sys.argv
backend = args[0] if args else "template"
n_per = int(args[1]) if len(args) > 1 else 10
MAX_TOTAL_WAIT = 4 * 3600
SEED = load_config()["seed"]


def wait_seconds(reason):
    """Seconds the provider asks us to wait ('try again in 6m49.5s'), or None."""
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", reason or "")
    return None if not m else int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))


pipe = Pipeline.load()
demo = pd.read_parquet(path("final") / "demo_messages.parquet")
picks = demo[demo.in_demo_pick]
picks = picks.assign(correct=picks.pred == picks["class"])
# stratify by class and by outcome so that misclassified / low-confidence cases are represented
parts = []
for c, g in picks.groupby("class"):
    wrong = g[~g.correct]
    wrong = wrong.sample(min(len(wrong), n_per // 2), random_state=SEED)
    right = g[g.correct]
    parts += [wrong, right.sample(min(len(right), n_per - len(wrong)), random_state=SEED)]
sample = pd.concat(parts)

out_dir = path("reports")
rows, streak, waited = [], 0, 0.0
with open(out_dir / f"reports_{backend}.jsonl", "w", encoding="utf-8") as fh:
    for i, m in enumerate(sample.itertuples()):
        veh = demo[demo.senderPseudo == m.senderPseudo][["sendTime", "sender", "senderPseudo", "messageID", "class"] + RAW_FEATURES]
        t0 = time.time(); res = pipe.analyse(veh, m.messageID); t1 = time.time()
        while True:
            ta = time.time()
            rep = generate_report(res["evidence"], backend=backend)
            tb = time.time()
            kind = None if rep.fallback_reason is None else ("validation" if "validation" in rep.fallback_reason else "unavailable")
            wait = wait_seconds(rep.fallback_reason) if kind == "unavailable" else None
            if patient and wait is not None and wait <= 3600 and waited + wait <= MAX_TOTAL_WAIT:
                print(f"  [{i+1}/{len(sample)}] provider limit reached: waiting {wait/60:.1f} min", flush=True)
                time.sleep(wait + 30); waited += wait + 30
                continue
            break
        if kind == "unavailable":                      # an outage is not a report result: never counted, never written
            streak += 1
            print(f"  [{i+1}/{len(sample)}] backend unavailable: {rep.fallback_reason[:150]}", flush=True)
            if streak >= 3 or "per day" in rep.fallback_reason or "TPD" in rep.fallback_reason:
                print("Stopping: provider limit (daily quota or repeated outage). Re-run later (add --patient to wait); finished reports are cached.")
                break
            time.sleep(30)
            continue
        streak = 0
        fh.write(json.dumps({"true_class": int(demo.loc[m.Index, "class"]), "evidence": res["evidence"], "report": rep.to_dict()}) + "\n")
        rows.append({"fallback_kind": kind, "llm_attempts": rep.attempts, "requested_backend": backend, "writer_used": rep.backend, "model": rep.model,
                     "passed": rep.validation["passed"], **{f"check_{k}": v for k, v in rep.validation["checks"].items()},
                     "fallback": rep.fallback_reason is not None, "correct_prediction": bool(m.correct),
                     "confidence_level": res["evidence"]["prediction"]["confidence_level"],
                     "words": len(rep.text.split()), "analyse_s": t1 - t0, "report_s": tb - ta})
        if backend != "template" and not rep.from_cache:
            time.sleep(30)                             # pacing for free-tier tokens-per-minute limits

df = pd.DataFrame(rows)
df.to_csv(out_dir / f"report_eval_{backend}.csv", index=False)
llm = df.writer_used != "template"
summ = {"requested_backend": backend, "model": df.model[llm].iloc[0] if llm.any() else "deterministic-template",
        "n_reports": len(df), "written_by_llm": int(llm.sum()),
        "llm_passed_first_try": int((llm & (df.llm_attempts == 1)).sum()), "llm_passed_after_retry": int((llm & (df.llm_attempts == 2)).sum()),
        "llm_rejected_twice_fallback_to_template": int((df.fallback_kind == "validation").sum()),
        "validation_pass_rate": round(df.passed.mean(), 4), "mean_words": round(df.words.mean(), 1),
        "median_analyse_s (features+model+SHAP)": round(df.analyse_s.median(), 3), "median_report_s": round(df.report_s.median(), 3),
        "misclassified_in_sample": int((~df.correct_prediction).sum()), "low_or_moderate_confidence_in_sample": int((df.confidence_level != "high").sum())}
f = path("results") / "report_faithfulness_summary.csv"
old = pd.read_csv(f) if f.exists() else pd.DataFrame()
old = old[old.requested_backend != backend] if len(old) else old
pd.concat([old, pd.DataFrame([summ])]).to_csv(f, index=False)
print(json.dumps(summ, indent=1))
