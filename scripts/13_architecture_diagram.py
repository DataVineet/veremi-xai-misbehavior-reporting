"""Step 13 - block diagram of the whole pipeline (for the report and the presentation).

Every box is a component that exists in this repository; nothing is drawn that is not implemented.

Run:  python scripts/13_architecture_diagram.py
Output: outputs/figures/architecture.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "architecture.png"

COLORS = {"data": "#DCE6F2", "ml": "#CFE8D5", "xai": "#FBE3B5", "llm": "#E3D5EE", "safe": "#F6CFCF", "ui": "#E8E8E8"}
W, H, DX, DY, Y0 = 3.35, 1.45, 3.8, 2.6, 0.85          # box size, column / row pitch, y of the bottom row

# (column, row, key, title, detail)
boxes = [
    (0, 1, "data", "VeReMi Extension", "simulated V2X messages\n20 classes, 1.2 GB CSV"),
    (1, 1, "data", "Sender-disjoint split", "70 / 15 / 15 by vehicle\n(no vehicle in two splits)"),
    (2, 1, "ml", "Feature engineering", "41 causal features: per-message,\nhistory per pseudonym, road map"),
    (3, 1, "ml", "XGBoost detector", "20-class classifier\nprediction + confidence level"),
    (3, 0, "xai", "SHAP (TreeSHAP)", "exact, additive contributions\nof the predicted class"),
    (2, 0, "xai", "Evidence package", "JSON: prediction, alternatives,\nSHAP top features, values, limits"),
    (1, 0, "llm", "Report writer", "LLM (OpenAI-compatible API)\nor deterministic template"),
    (0, 0, "safe", "Faithfulness validator", "checks numbers, classes, features;\n1 retry, then template fallback"),
]
fig, ax = plt.subplots(figsize=(15, 6.6))
ax.set_xlim(-0.2, 3 * DX + W + 0.2); ax.set_ylim(-0.3, 6.5); ax.axis("off")
pos = {}
for col, row, kind, title, detail in boxes:
    x, y = col * DX, row * DY + Y0
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.12", fc=COLORS[kind], ec="#444", lw=1.1))
    ax.text(x + W / 2, y + H - 0.32, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(x + W / 2, y + 0.5, detail, ha="center", va="center", fontsize=9.2, linespacing=1.35)
    pos[(col, row)] = (x, y)


def arrow(a, b, dir_):
    (x1, y1), (x2, y2) = pos[a], pos[b]
    if dir_ == "right":
        p, q = (x1 + W, y1 + H / 2), (x2, y2 + H / 2)
    elif dir_ == "left":
        p, q = (x1, y1 + H / 2), (x2 + W, y2 + H / 2)
    else:  # down
        p, q = (x1 + W / 2, y1), (x2 + W / 2, y2 + H)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333"))


for a, b in [((0, 1), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (3, 1))]:
    arrow(a, b, "right")
arrow((3, 1), (3, 0), "down")
for a, b in [((3, 0), (2, 0)), ((2, 0), (1, 0)), ((1, 0), (0, 0))]:
    arrow(a, b, "left")

x0, y0 = pos[(0, 0)]
FULL = 3 * DX + W
ax.add_patch(FancyBboxPatch((x0, -0.1), FULL, 0.55, boxstyle="round,pad=0.02,rounding_size=0.1", fc=COLORS["ui"], ec="#444", lw=1.1))
ax.text(x0 + FULL / 2, 0.175, "Validated human-readable report  →  Streamlit app (detect, explain, report, model performance, limitations)",
        ha="center", va="center", fontsize=11, fontweight="bold")
ax.add_patch(FancyArrowPatch((x0 + W / 2, y0), (x0 + W / 2, 0.47), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333"))

ax.text(0, 6.25, "ML detects  ·  SHAP explains  ·  the LLM only reports  ·  a validator checks every report against the evidence",
        fontsize=12, style="italic", va="center")
handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[k], ec="#444") for k in ("data", "ml", "xai", "llm", "safe")]
ax.legend(handles, ["data", "detector (ML)", "explainer (XAI)", "reporter (LLM)", "safety check"], loc="upper left",
          bbox_to_anchor=(0.0, 0.955), ncol=5, frameon=False, fontsize=9.5)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("written", OUT)
