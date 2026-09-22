"""Step 6 (analysis) — number-to-feature binding check for LLM-written reports.

The validator guarantees that every number exists in the evidence, but not that it is attached to the right
feature. This offline check looks at every report line that names exactly ONE top-contribution feature and flags
the line if it contains another feature's SHAP/value while not containing its own. No API calls.

Run:  python scripts/06_binding_check.py groq_small
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from veremi_xai.config import path
from veremi_xai.reporting.validator import NUM, normalise

backend = sys.argv[1] if len(sys.argv) > 1 else "groq"
lines_checked = suspicious = reports = 0
for raw in open(path("reports") / f"reports_{backend}.jsonl", encoding="utf-8"):
    d = json.loads(raw)
    if d["report"]["backend"] == "template":
        continue
    reports += 1
    top = {c["feature"]: c for c in d["evidence"]["shap"]["top_contributions"]}
    for line in normalise(d["report"]["text"]).splitlines():
        named = [f for f in top if re.search(rf"`{re.escape(f)}`", line)]
        if len(named) != 1:
            continue
        f = named[0]
        nums = {abs(float(t.rstrip("% "))) for t in NUM.findall(line)}
        own = {abs(v) for v in (top[f]["value"], top[f]["shap"]) if v is not None}
        others = {abs(v) for g, c in top.items() if g != f for v in (c["value"], c["shap"]) if v is not None} - own
        lines_checked += 1
        if nums & others and not nums & own:
            suspicious += 1
            print("SUSPICIOUS:", line.strip()[:200])
result = {"backend": backend, "llm_reports": reports, "single_feature_lines_checked": lines_checked, "suspicious_lines": suspicious}
print(json.dumps(result))
out = path("results") / "binding_check.json"
allres = json.loads(out.read_text()) if out.exists() else {}
allres[backend] = result
out.write_text(json.dumps(allres, indent=1))
