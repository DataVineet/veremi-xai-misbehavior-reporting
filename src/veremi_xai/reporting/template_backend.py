"""Deterministic, offline reporter. Always available; also the faithfulness baseline for LLM backends."""
from __future__ import annotations

ACTIONS = {
    "genuine": "No action required. Keep the message in the normal processing path.",
    "fault": "Treat the sender's data as unreliable. Flag the vehicle for sensor/OBU diagnostics rather than as an adversary, and confirm over its next messages.",
    "attack": "Escalate to the misbehavior authority with this evidence package, down-weight or ignore this identity's data in safety applications, and confirm over its next messages.",
}


def render(ev: dict) -> str:
    p, s = ev["prediction"], ev["shap"]
    alt = p["alternatives"][0]
    lines = ["## Summary",
             f"The detector classified message {ev['message']['messageID']} (pseudonym {ev['message']['senderPseudo']}) as "
             f"**{p['class_name']}** (category: {p['category']}) with confidence {p['confidence']} ({p['confidence_level']}). "
             f"Definition: {p['definition']}", "", "## Observed evidence"]
    for c in s["top_contributions"]:
        val = "n/a (no earlier message)" if c["value"] is None else c["value"]
        lines.append(f"- `{c['feature']}` = {val} — {c['description']}. SHAP {c['shap']} ({c['effect']}).")
    lines += ["", "## Model interpretation",
              f"These {len(s['top_contributions'])} features account for {s['share_of_total_attribution_in_top']} of the total absolute SHAP attribution "
              f"for the class {s['explained_class']}. Starting from a base value of {s['base_value']}, the contributions add up to a model output of "
              f"{s['model_output']} in the {s['output_space']}. The model relied mainly on `{s['top_contributions'][0]['feature']}`.",
              "", "## Uncertainty and limitations",
              f"Confidence level is {p['confidence_level']}. The next most likely class is {alt['class_name']} with probability {alt['probability']}."]
    if not ev["message"]["has_history"]:
        lines.append("This is the first message seen from this identity, so history-based features were not available.")
    lines += [f"- {x}" for x in ev["limitations"][:3]]
    lines += ["", "## Suggested analyst action", ACTIONS[p["category"]]]
    return "\n".join(lines)
