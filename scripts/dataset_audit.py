"""Dataset audit (read-only) — reproduces every [EXP] number in docs/background.md §3–4.

Run from the project root with the project venv:
    .venv\Scripts\python.exe scripts\dataset_audit.py        (Windows)
Needs ~1.5 GB free RAM; runs in well under a minute. Writes nothing to disk.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

pd.set_option("display.width", 250); pd.set_option("display.max_columns", None); pd.set_option("display.max_rows", 200)
P = Path(__file__).resolve().parents[1] / "data" / "mixalldata_clean.csv"

# ===================== PART 1: structure, gaps, class signatures, duplicate vectors =====================
t0 = time.time()
cols = ["sendTime","sender","senderPseudo","class","posx","posy","spdx","spdy","aclx","acly","hedx","hedy","posx_n","spdx_n","hedx_n"]
dt = {c: "float64" for c in cols}; dt.update({"sender":"int32","senderPseudo":"int64","class":"int8"})
df = pd.read_csv(P, usecols=cols, dtype=dt, engine="pyarrow")
print("loaded", df.shape, f"{time.time()-t0:.0f}s", f"{df.memory_usage().sum()/1e6:.0f} MB")
df["row"] = np.arange(len(df))

# 1. file layout: are sender rows contiguous? time-sorted in file order?
s = df["sender"].to_numpy(); runs = 1 + int((s[1:] != s[:-1]).sum())
print("\n[layout] sender runs in file:", runs, "| unique senders:", df["sender"].nunique())
same = s[1:] == s[:-1]; dtf = np.diff(df["sendTime"].to_numpy())
print("[layout] consecutive same-sender rows with non-increasing time in FILE order:", int((dtf[same] <= 0).sum()))
c = df["class"].to_numpy(); print("[layout] class runs in file:", 1 + int((c[1:] != c[:-1]).sum()))

# 2. sender/class structure
g = df.groupby("sender")
per = g.agg(cls=("class","first"), ncls=("class","nunique"), n=("row","size"), npseudo=("senderPseudo","nunique"),
            t0=("sendTime","min"), t1=("sendTime","max"),
            posx_sd=("posx","std"), posy_sd=("posy","std"))
print("\n[sender] multi-class senders:", int((per.ncls > 1).sum()))
print("[sender] rows per sender:\n", per.n.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).round(1).to_string())

# 3. true within-sender gaps (sorted by sender,time), incl. non-positive
d = df.sort_values(["sender","sendTime"], kind="stable")
gap = d["sendTime"].diff().to_numpy().copy(); first = d["sender"].to_numpy() != np.roll(d["sender"].to_numpy(), 1); first[0] = True
gap[first] = np.nan; d["gap"] = gap
gv = gap[~np.isnan(gap)]
print("\n[gaps] n:", len(gv), "| ==0:", int((gv==0).sum()), "| <0:", int((gv<0).sum()), "| >1.0001:", int((gv>1.0001).sum()), "| max:", gv.max())
print("[gaps] quantiles:", np.quantile(gv,[0,.01,.05,.25,.5,.75,.95,.99,1]).round(4))

# 4. per-class signature table
d["spd"] = np.hypot(d["spdx"], d["spdy"]); d["acl"] = np.hypot(d["aclx"], d["acly"])
d["dpos"] = np.hypot(d["posx"].diff(), d["posy"].diff()); d.loc[first, "dpos"] = np.nan
d["zero_pos"] = (d["posx"] == 0) & (d["posy"] == 0)
d["zero_noise"] = (d["posx_n"] == 0)
sig = d.groupby("class").agg(rows=("row","size"), gap_med=("gap","median"), gap_mean=("gap","mean"),
        gap_lt1=("gap", lambda x: np.nanmean(x < 0.999)), spd_mean=("spd","mean"), spd_sd=("spd","std"),
        spd_eq0=("spd", lambda x: (x == 0).mean()), acl_mean=("acl","mean"),
        dpos_med=("dpos","median"), dpos_p95=("dpos", lambda x: np.nanquantile(x,.95)),
        dpos_eq0=("dpos", lambda x: np.nanmean(x == 0)), zero_pos=("zero_pos","mean"), zero_noise=("zero_noise","mean"))
ps = per.groupby("cls").agg(senders=("n","size"), rows_per_sender_med=("n","median"), npseudo_med=("npseudo","median"),
        npseudo_mean=("npseudo","mean"), npseudo_max=("npseudo","max"), dur_med=("t1", lambda x: 0),
        posx_sd_med=("posx_sd","median"))
ps["dur_med"] = (per.t1 - per.t0).groupby(per.cls).median()
print("\n[class signature — message level]\n", sig.round(3).to_string())
print("\n[class signature — sender level]\n", ps.round(2).to_string())

# within-sender std of speed per class (constant-speed signature)
ws = d.groupby("sender")["spd"].std().groupby(per.cls).median()
print("\n[within-sender speed std, median by class]\n", ws.round(3).to_string())

# 5. pseudonym 1
p1 = df[df.senderPseudo == 1]
print("\n[pseudo==1] rows:", len(p1), "senders:", p1.sender.nunique(), "classes:", p1["class"].value_counts().to_dict())
print("[pseudo] senderPseudo min/max:", df.senderPseudo.min(), df.senderPseudo.max())
ex = df[df.senderPseudo != 1].head(3)[["sender","senderPseudo"]]; print(ex.to_string())

# 6. feature-vector duplicates (behavioural features only) & cross-class collisions
feat = ["posx","posy","spdx","spdy","aclx","acly","hedx","hedy"]
h = pd.util.hash_pandas_object(df[feat], index=False).to_numpy()
hs = pd.DataFrame({"h": h, "cls": df["class"].to_numpy(), "sender": df["sender"].to_numpy()})
dup = hs[hs.duplicated("h", keep=False)]
print("\n[dups] rows whose 8-feature vector occurs >1x:", len(dup), f"({len(dup)/len(df):.2%})")
print("[dups] by class (share of class rows):\n", (dup.cls.value_counts().sort_index() / df["class"].value_counts().sort_index()).round(3).to_string())
gg = dup.groupby("h").agg(ncls=("cls","nunique"), nsend=("sender","nunique"))
print("[dups] distinct dup vectors:", len(gg), "| spanning >1 sender:", int((gg.nsend>1).sum()), "| spanning >1 class:", int((gg.ncls>1).sum()))
x = dup.merge(gg[gg.ncls>1], left_on="h", right_index=True)
print("[dups] rows involved in cross-class identical vectors, by class:\n", x.cls.value_counts().sort_index().to_string())
print(f"\ndone {time.time()-t0:.0f}s")

# ===================== PART 2: time coverage, zero rows, noise columns, pseudonym-level gaps =====================
df = pd.read_csv(P, engine="pyarrow")
print("dtypes:", df.dtypes.value_counts().to_dict(), "| NaN total:", int(df.isna().sum().sum()), "| inf:", int(np.isinf(df.select_dtypes("float")).sum().sum()))
# hour-of-day coverage
h = (df.sendTime // 3600).astype(int)
print("\n[hour histogram, rows]\n", h.value_counts().sort_index().to_dict())
print("[hours with attackers]", sorted(h[df["class"]>0].unique().tolist()))
# class 12 zero rows: where in sequence?
for c in (12, 9):
    s = df[df["class"]==c]
    sid = s.sender.value_counts().index[5]
    one = s[s.sender==sid][["sendTime","senderPseudo","posx","posy","posx_n","spdx","spdy","hedx","hedx_n"]]
    print(f"\n[class {c}] example sender {sid}, n={len(one)}\n", one.head(8).round(3).to_string(), "\n ...\n", one.tail(4).round(3).to_string())
# all-zero rows by class
z = (df[["posx","posy","spdx","spdy","aclx","acly","hedx","hedy","posx_n","hedx_n"]]==0).all(axis=1)
print("\n[all-zero kinematic rows by class]", df.loc[z,"class"].value_counts().sort_index().to_dict())
# heading: unit vector? noise columns semantics
print("\n[heading norm] quantiles:", np.hypot(df.hedx, df.hedy).quantile([0,.01,.5,.99,1]).round(4).tolist())
print("[noise cols describe]\n", df[["posx_n","spdx_n","aclx_n","hedx_n","hedy_n"]].describe().round(4).to_string())
print("[spdx_n sign] share negative:", (df.spdx_n<0).mean().round(3), "| aclx_n negative:", (df.aclx_n<0).mean().round(3))
# gaps grouped by senderPseudo (receiver-observable identity) for DoS/sybil classes
d = df.sort_values(["senderPseudo","sendTime"], kind="stable")
g = d.sendTime.diff().to_numpy().copy(); first = d.senderPseudo.to_numpy() != np.roll(d.senderPseudo.to_numpy(),1); first[0]=True; g[first]=np.nan
d["pgap"]=g
print("\n[median gap per PSEUDONYM by class]\n", d.groupby("class").pgap.median().round(3).to_dict())
print("[rows per pseudonym, median by class]\n", d.groupby(["class","senderPseudo"]).size().groupby("class").median().to_dict())
# sender id vs pseudo relation & sender id ranges per class (is sender id informative?)
print("\n[sender id range by class]\n", df.groupby("class").sender.agg(["min","max"]).T.to_string())
# messageID monotonic w/ time?
print("\n[messageID] corr with sendTime:", round(df.messageID.corr(df.sendTime),4))
