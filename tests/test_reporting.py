"""Reporting-layer tests (no network, no API key needed). Run: .venv/Scripts/python.exe -m pytest -q  (or run this file directly)

A local mock OpenAI-compatible server checks the real HTTP code path, the faithfulness validator,
the corrective retry and the template fallback.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from veremi_xai.reporting import template_backend
from veremi_xai.reporting.openai_compat import OpenAICompatibleBackend
from veremi_xai.reporting.validator import validate

EV = {
    "schema_version": "1.0", "evidence_hash": "testhash00000000",
    "message": {"messageID": 123456, "sendTime_s": 50000.25, "senderPseudo": 10987, "has_history": True},
    "prediction": {"class_id": 13, "class_name": "DoS", "category": "attack", "definition": "Denial of service: the vehicle sends messages at a frequency higher than the limit set by the standard.",
                   "confidence": 0.9731, "confidence_level": "high",
                   "alternatives": [{"class_id": 0, "class_name": "Genuine", "probability": 0.0201}, {"class_id": 15, "class_name": "DoS disruptive", "probability": 0.0041}]},
    "shap": {"explained_class": "DoS", "output_space": "raw margin (log-odds score before softmax) of the explained class", "base_value": -0.412, "model_output": 7.934,
             "top_contributions": [
                 {"rank": 1, "feature": "r_dt_mean", "description": "mean time between the last messages of this identity (s)", "value": 0.25, "shap": 4.211, "effect": "pushes towards the predicted class"},
                 {"rank": 2, "feature": "dt", "description": "time since the previous message of the same identity (s); genuine beacons arrive every 1 s", "value": 0.25, "shap": 2.876, "effect": "pushes towards the predicted class"},
                 {"rank": 3, "feature": "road_dist", "description": "distance to genuine traffic (m)", "value": 1.2, "shap": -0.31, "effect": "pushes away from the predicted class"}],
             "share_of_total_attribution_in_top": 0.871},
    "transmitted_values": {"posx": 512.337, "posy": 730.118, "spdx": 3.2, "spdy": -8.1, "aclx": 0.1, "acly": 0.2, "hedx": 0.37, "hedy": -0.93, "spd": 8.709, "acl": 0.224},
    "model": {"name": "m", "version": "1", "algorithm": "xgb", "n_features": 41, "identity": "senderPseudo", "test_macro_f1": 0.9, "test_accuracy": 0.95},
    "limitations": ["Simulated data."],
}
GOOD = """## Summary
The model classified this message as DoS with confidence 0.9731 (97.31%), which is high.
## Observed evidence
`r_dt_mean` is 0.25 s and `dt` is 0.25 s, while genuine beacons arrive every 1 s.
## Model interpretation
The model relied on `r_dt_mean` (SHAP 4.211) and `dt` (SHAP 2.876); `road_dist` pushed away (-0.31).
## Uncertainty and limitations
The next class is Genuine at 0.0201. Simulated data.
## Suggested analyst action
Escalate."""
BAD_NUMBER = GOOD.replace("0.25 s and", "0.47 s and")
BAD_CLASS = GOOD.replace("Escalate.", "This may be Grid Sybil.")
BAD_OVERRIDE = GOOD.replace("Escalate.", "I believe the message is actually a benign one.")
BAD_FEATURE = GOOD.replace("Escalate.", "Also `spd_resid` was large.")


TYPOGRAPHIC = GOOD.replace("(97.31%)", "(97.31 %)").replace("(-0.31)", "(\u20110.31)").replace("0.0201", "2.01e\u201102")


REAL_LLM_STYLE = (GOOD.replace("## Observed evidence\n", "## Observed evidence\nMessage 123\u202f456 from pseudonym 10,987 (`senderPseudo`) sent at 50\u202f000.25 s.\n")
                  .replace("Escalate.", "Flag the pseudonym as a likely source of random position offsets. | --- | --- |"))


def test_validator_handles_real_llm_style():
    # thousands separators (thin space / comma), back-ticked JSON key, lower-case descriptive phrase equal to a class name, table syntax
    v = validate(REAL_LLM_STYLE, EV)
    assert v["passed"], v
    sci = dict(EV, transmitted_values=dict(EV["transmitted_values"], aclx=1.43e-12))
    assert validate(GOOD.replace("Escalate.", "`aclx` was 1.43\u202f×\u202f10⁻¹²."), sci)["passed"]


def test_validator_handles_llm_typography():
    # "97.31 %" with a space, non-breaking hyphen as minus sign and inside a scientific-notation exponent
    assert validate(TYPOGRAPHIC, EV)["passed"], validate(TYPOGRAPHIC, EV)


def test_template_always_valid():
    assert validate(template_backend.render(EV), EV)["passed"]


def test_validator_accepts_faithful_and_rejects_unfaithful():
    assert validate(GOOD, EV)["passed"], validate(GOOD, EV)
    for bad, key in ((BAD_NUMBER, "numbers"), (BAD_CLASS, "prediction"), (BAD_OVERRIDE, "no_override"), (BAD_FEATURE, "features")):
        v = validate(bad, EV)
        assert not v["passed"] and not v["checks"][key], (key, v)


class _Handler(BaseHTTPRequestHandler):
    replies = []
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["temperature"] == 0.0 and body["messages"][0]["role"] == "system"
        text = _Handler.replies.pop(0)
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a):
        pass


def test_http_backend_retry_and_fallback(monkeypatch=None):
    import veremi_xai.reporting as rp
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    be = OpenAICompatibleBackend("mock", f"http://127.0.0.1:{srv.server_port}/v1", "mock-model", None)
    _Handler.replies = [GOOD]
    assert be.generate(EV) == GOOD
    orig = rp._profile
    rp._profile = lambda name: be
    try:
        _Handler.replies = [BAD_NUMBER, GOOD]                       # corrective retry succeeds
        r = rp.generate_report(EV, backend="mock", use_cache=False)
        assert r.backend == "mock" and r.validation["passed"]
        _Handler.replies = [BAD_NUMBER, BAD_CLASS]                  # fails twice -> safe template fallback
        r = rp.generate_report(EV, backend="mock", use_cache=False)
        assert r.backend == "template" and r.validation["passed"] and r.rejected_llm_text and "validation" in r.fallback_reason
        srv.shutdown()
        be.base_url = "http://127.0.0.1:9/v1"; be.timeout_s = 1      # unreachable -> fallback
        import veremi_xai.reporting.openai_compat as oc
        oc.time.sleep = lambda s: None
        r = rp.generate_report(EV, backend="mock", use_cache=False)
        assert r.backend == "template" and "unavailable" in r.fallback_reason
    finally:
        rp._profile = orig


if __name__ == "__main__":
    test_validator_handles_real_llm_style(); test_validator_handles_llm_typography(); test_template_always_valid(); test_validator_accepts_faithful_and_rejects_unfaithful(); test_http_backend_retry_and_fallback()
    print("all reporting tests passed")
