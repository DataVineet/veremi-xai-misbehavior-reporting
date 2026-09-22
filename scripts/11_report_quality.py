"""Step 11 - completeness of the generated reports (objective rules; no LLM judge and no human rating).

For every stored report set (outputs/reports/reports_<writer>.jsonl) it checks that the text
  * names the most important SHAP feature,             * states the confidence level given in the evidence,
  * names the runner-up class when confidence is not high,   * names the predicted class in the Summary,
  * mentions that the data is simulated (limitation).
Faithfulness (no invented facts) is checked separately by the validator; this script only measures completeness.

Run:  python scripts/11_report_quality.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from veremi_xai.config import path

rows = []
for f in sorted(path("reports").glob("reports_*.jsonl")):
    writer = f.stem.replace("reports_", "")
    recs = [json.loads(l) for l in open(f, encoding="utf-8")]
    recs = [r for r in recs if writer == "template" or r["report"]["backend"] != "template"]        # only texts really written by that writer
    if not recs:
        continue
    res = {"top_feature": [], "confidence_level": [], "runner_up": [], "class_in_summary": [], "simulated_caveat": [], "words": []}
    for r in recs:
        ev, text = r["evidence"], r["report"]["text"]; low = text.lower()
        summary = re.split(r"^##\s*Observed", text, flags=re.M | re.I)[0]
        lvl = ev["prediction"]["confidence_level"]
        res["top_feature"].append(ev["shap"]["top_contributions"][0]["feature"] in text)
        res["confidence_level"].append(re.search(rf"\b{lvl}\b", low) is not None)
        if lvl != "high":
            res["runner_up"].append(ev["prediction"]["alternatives"][0]["class_name"].lower() in low)
        res["class_in_summary"].append(ev["prediction"]["class_name"].lower() in summary.lower())
        res["simulated_caveat"].append("simulat" in low)
        res["words"].append(len(text.split()))
    pct = lambda v: round(100 * sum(v) / len(v), 1)
    rows.append({"writer": writer, "model": recs[0]["report"]["model"], "reports": len(recs), "names_top_shap_feature_%": pct(res["top_feature"]),
                 "states_confidence_level_%": pct(res["confidence_level"]), "names_runner_up_when_not_high_%": pct(res["runner_up"]), "reports_with_non_high_confidence": len(res["runner_up"]),
                 "class_named_in_summary_%": pct(res["class_in_summary"]), "mentions_simulated_data_%": pct(res["simulated_caveat"]), "mean_words": round(sum(res["words"]) / len(res["words"]), 1)})
df = pd.DataFrame(rows)
df.to_csv(path("results") / "report_quality_summary.csv", index=False)
print(df.to_string(index=False))
