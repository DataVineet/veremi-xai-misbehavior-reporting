"""Shared chart styling (one palette for matplotlib figures and the Streamlit app).

Colour is assigned by job: single blue hue for magnitude, fixed categorical slots for identity
(category genuine/fault/attack; split type), blue-vs-orange for SHAP polarity (always with a text label too).
"""
import matplotlib as mpl

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ_BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CATEGORY_COLORS = {"genuine": BLUE, "fault": ORANGE, "attack": AQUA}
POLARITY = {"pushes towards the predicted class": BLUE, "pushes away from the predicted class": ORANGE}


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
