"""End-to-end tests: pipeline (features -> model -> SHAP -> evidence -> report) and the Streamlit app (headless).

Needs the deployable artifacts in models/ (created by scripts/03_finalize_and_explain.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from veremi_xai.config import RAW_FEATURES, path
from veremi_xai.pipeline import Pipeline
from veremi_xai.reporting import generate_report

COLS = ["sendTime", "sender", "senderPseudo", "messageID", "class"] + RAW_FEATURES


def test_pipeline_end_to_end():
    pipe = Pipeline.load()
    demo = pd.read_parquet(path("final") / "demo_messages.parquet")
    picks = pd.concat([demo[demo.in_demo_pick].groupby("class").sample(2, random_state=3), demo[demo.in_demo_pick & (demo.senderPseudo == 1)].sample(3, random_state=3)])
    for m in picks.itertuples():
        res = pipe.analyse(demo.loc[demo.senderPseudo == m.senderPseudo, COLS], m.messageID)
        ev, c = res["evidence"], res["evidence"]["prediction"]["class_id"]
        # the app reproduces the batch prediction stored at finalisation time
        assert c == m.pred and abs(res["proba"][c] - m.conf) < 1e-4
        # SHAP is exactly additive w.r.t. the deployed model
        margin = pipe.explainer.margins(res["history"][res["history"].messageID == m.messageID])[0]
        assert np.abs(res["shap_row"].sum(0) + pipe.explainer.base_values - margin).max() < 1e-3
        assert abs(ev["shap"]["model_output"] - margin[c]) < 2e-3
        assert "class" not in ev["message"] and "sender" not in ev["message"]          # no ground truth / hidden identity in the evidence
        rep = generate_report(ev, backend="template")
        assert rep.validation["passed"], rep.validation


def test_streamlit_app_runs_and_generates_report():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    at.sidebar.selectbox[0].set_value("template").run()          # tests must not need the network or spend API quota
    next(b for b in at.button if "Generate report" in b.label).click().run()
    assert not at.exception, at.exception
    assert any("## Summary" in m.value for m in at.markdown), "report not rendered"
    assert any("validation passed" in s.value for s in at.success)


if __name__ == "__main__":
    test_pipeline_end_to_end(); print("pipeline ok")
    test_streamlit_app_runs_and_generates_report(); print("app ok")
