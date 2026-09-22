"""Quick check of the LLM report writer: shows which backends are usable and writes ONE report.

Run:  python scripts/check_llm.py            (auto: first profile whose key is in .env, else template)
      python scripts/check_llm.py groq       (force a profile from config.yaml -> llm.profiles)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # Windows consoles default to cp1252; LLM text may contain any Unicode

from veremi_xai.config import RAW_FEATURES, load_config, path
from veremi_xai.pipeline import Pipeline
from veremi_xai.reporting import available_backends, generate_report

backend = sys.argv[1] if len(sys.argv) > 1 else None
print("Backends usable right now:", available_backends())
if backend and backend != "template":
    print("Profile:", load_config()["llm"]["profiles"][backend])

pipe = Pipeline.load()
demo = pd.read_parquet(path("final") / "demo_messages.parquet")
m = demo[demo.in_demo_pick & (demo["class"] == 13) & (demo.pred == 13)].iloc[40]          # a correctly detected DoS message
cols = ["sendTime", "sender", "senderPseudo", "messageID", "class"] + RAW_FEATURES
ev = pipe.analyse(demo.loc[demo.senderPseudo == m.senderPseudo, cols], m.messageID)["evidence"]
rep = generate_report(ev, backend=backend, use_cache=False)

print("\n" + "=" * 90 + f"\nWRITTEN BY: {rep.backend}  (model: {rep.model})")
if rep.fallback_reason:
    print("!! FELL BACK TO TEMPLATE because:", rep.fallback_reason)
print("=" * 90 + "\n" + rep.text + "\n" + "=" * 90)
print("Faithfulness validation passed:", rep.validation["passed"], "|", rep.validation["checks"])
if rep.validation["violations"]:
    print("Violations:", rep.validation["violations"])
if rep.rejected_llm_text:
    print("\n--- rejected LLM draft ---\n" + rep.rejected_llm_text)
