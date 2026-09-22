"""Class-mapping verification (read-only): positive, data-based tests for every class identity.

Complements scripts/dataset_audit.py. Each test targets a defining property taken from the
VeReMi Extension paper's attack definitions. Run from project root with the project venv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
P = Path(__file__).resolve().parents[1] / "data" / "mixalldata_clean.csv"
cols = ["sendTime", "sender", "senderPseudo", "class", "posx", "posy", "spdx", "spdy", "aclx", "acly", "hedx", "hedy"]
df = pd.read_csv(P, usecols=cols, engine="pyarrow")           # file is already sender-contiguous & time-sorted
new = df.sender.ne(df.sender.shift())                           # first row of each sender block
df["dt"] = df.sendTime.diff().where(~new)
df["dpos"] = np.hypot(df.posx.diff(), df.posy.diff()).where(~new)
df["spd"] = np.hypot(df.spdx, df.spdy)
df["v_implied"] = df.dpos / df.dt                               # speed implied by reported positions
df["resid"] = df.spd - df.v_implied                             # reported speed minus position-implied speed
df["zero"] = (df[["posx", "posy", "spdx", "spdy", "hedx", "hedy"]] == 0).all(axis=1)

# T1 — position/speed consistency. Const offset keeps kinematics consistent; random offset breaks them.
g = df[~df.zero].groupby("sender")
per = pd.DataFrame({"cls": g["class"].first(), "resid_mean": g.resid.mean(), "resid_sd": g.resid.std(),
                    "spd_sd": g.spd.std(), "abs_resid_med": g.resid.apply(lambda x: x.abs().median())})
t1 = per.groupby("cls").agg(resid_mean_med=("resid_mean", "median"), resid_mean_abs_med=("resid_mean", lambda x: x.abs().median()),
                            resid_sd_med=("resid_sd", "median"), abs_resid_med=("abs_resid_med", "median"), spd_sd_med=("spd_sd", "median"))
print("[T1] per-sender (reported speed - position-implied speed): median over senders\n", t1.round(3).to_string())

# T2 — off-road test. Cells (5 m) ever occupied by a genuine vehicle = 'road'. Share of rows outside road cells.
cell = (np.floor(df.posx / 5).astype("int64") * 100000 + np.floor(df.posy / 5).astype("int64"))
gen_senders = df.loc[df["class"] == 0, "sender"].unique()
rng = np.random.default_rng(42); held = set(rng.choice(gen_senders, size=len(gen_senders) // 10, replace=False).tolist())
is_held = df.sender.isin(held)
road = set(cell[(df["class"] == 0) & ~is_held].unique().tolist())
off = ~cell.isin(road)
res = off[~df.zero].groupby(df["class"][~df.zero]).mean()
print("\n[T2] share of rows in 5 m cells never visited by (non-held-out) genuine vehicles, by class")
print("     held-out genuine baseline:", round(off[is_held].mean(), 4)); print(res.round(4).to_string())

# T3 — replay provenance. Match each attack row's kinematic vector to genuine rows of OTHER senders.
feat = ["posx", "posy", "spdx", "spdy", "aclx", "acly", "hedx", "hedy"]
h = pd.util.hash_pandas_object(df[feat], index=False)
src = pd.DataFrame({"h": h[df["class"] == 0], "src": df.sender[df["class"] == 0]}).drop_duplicates("h").set_index("h").src
df["src"] = h.map(src)
a = df[(df["class"] > 0) & ~df.zero]
ga = a.groupby("sender")
pr = pd.DataFrame({"cls": ga["class"].first(), "copied": ga.src.apply(lambda x: x.notna().mean()),
                   "n_src": ga.src.nunique(), "n": ga.size(),
                   "src_switch": ga.src.apply(lambda x: (x.dropna().ne(x.dropna().shift())).mean() if x.notna().sum() > 1 else np.nan)})
t3 = pr.groupby("cls").agg(copied_share_med=("copied", "median"), distinct_sources_med=("n_src", "median"),
                           msgs_med=("n", "median"), source_switch_rate_med=("src_switch", "median"))
print("\n[T3] share of a sender's messages that are exact copies of a genuine vehicle's message; #distinct source vehicles;"
      "\n     source_switch_rate = how often consecutive copied messages come from a different source\n", t3.round(3).to_string())

# T4 — self-repetition (DoS re-sends its own current state between simulation steps)
df["same_as_prev"] = (df[feat].eq(df[feat].shift()).all(axis=1)) & ~new
print("\n[T4] share of messages identical to the sender's previous message\n", df.groupby("class").same_as_prev.mean().round(3).to_string())

# T5 — pseudonym dynamics: how often the pseudonym changes between consecutive messages; does it change with the replay source?
df["pchg"] = df.senderPseudo.ne(df.senderPseudo.shift()) & ~new
print("\n[T5] pseudonym change rate between consecutive messages\n", df.groupby("class").pchg.mean().round(3)[lambda s: s > 0].to_string())
s17 = df[(df["class"] == 17) & df.src.notna()].copy()
s17["schg"] = s17.src.ne(s17.groupby("sender").src.shift()) & s17.groupby("sender").src.shift().notna()
print("[T5] class 17: P(pseudonym changes | source changes) =", round(s17.loc[s17.schg, "pchg"].mean(), 3),
      "| P(pseudonym changes | source same) =", round(s17.loc[~s17.schg, "pchg"].mean(), 3))

# T6 — random-field tests: uniformity of position over the playground & speed range
for c in (3, 7, 14, 18):
    s = df[df["class"] == c]
    print(f"[T6] class {c}: posx range {s.posx.min():.0f}..{s.posx.max():.0f} (sd {s.posx.std():.0f}; uniform would be {(s.posx.max()-s.posx.min())/12**.5:.0f}) | "
          f"spdx range {s.spdx.min():.1f}..{s.spdx.max():.1f} | lag-1 autocorr posx {s.posx.corr(s.posx.shift()):.3f}, spdx {s.spdx.corr(s.spdx.shift()):.3f}")

# T7 — delayed messages: does the reported state lag behind? Heading/speed are consistent, positions trail. Check leading zero-run length in seconds.
z = df[df["class"] == 12].groupby("sender").zero
lead = z.apply(lambda x: int((~x).to_numpy().argmax()) if (~x).any() else len(x))
print("\n[T7] class 12: leading all-zero run per sender (messages = seconds): median", lead.median(), "| share of senders with zero-run at start:", round((lead > 0).mean(), 3),
      "| zero rows NOT in the leading run:", int(df[(df['class'] == 12)].zero.sum() - lead.sum()))

# T8 — component-wise speed-offset test (classes 5-8 keep genuine positions, so d(pos)/dt is the true velocity).
# Constant offset -> (reported - true) is constant within a sender (low sd, non-zero mean); random offset -> high sd.
df["ex"] = df.spdx - (df.posx.diff() / df.dt).where(~new)
df["ey"] = df.spdy - (df.posy.diff() / df.dt).where(~new)
g8 = df[df["class"].isin([0, 5, 6, 7, 8])].groupby("sender")
p8 = pd.DataFrame({"cls": g8["class"].first(), "ex_mean": g8.ex.mean(), "ex_sd": g8.ex.std(), "ey_mean": g8.ey.mean(), "ey_sd": g8.ey.std()})
t8 = p8.groupby("cls").agg(ex_abs_mean_med=("ex_mean", lambda x: x.abs().median()), ex_sd_med=("ex_sd", "median"),
                           ey_abs_mean_med=("ey_mean", lambda x: x.abs().median()), ey_sd_med=("ey_sd", "median"))
print("\n[T8] per-sender error of reported velocity components vs position-implied velocity (median over senders)\n", t8.round(3).to_string())
ex6 = p8[p8.cls == 6].ex_mean; ex8 = p8[p8.cls == 8].ex_mean
print("[T8] class 6 per-sender mean x-offset quantiles:", ex6.quantile([.05, .25, .5, .75, .95]).round(2).tolist())
print("[T8] class 8 per-sender mean x-offset quantiles:", ex8.quantile([.05, .25, .5, .75, .95]).round(2).tolist())
