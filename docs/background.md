# Background, verified facts and methodology

**Project:** LLM-Assisted Explainable Misbehavior Reporting Framework for Cooperative Intelligent Transportation Systems (C-ITS)

This document collects the foundations of the project: the objective and design principles (§1), the dataset and what was verified about it (§2–3), the class-mapping evidence (§4), the methodological choices (§5) and the literature review (§6). Results and how to run the code are in the [README](../README.md); dated measurements are in [research_log.md](research_log.md); the design decisions that were actually built, each with its evidence, are in [architecture_decisions.md](architecture_decisions.md). §5 was written as the *pre-implementation plan*; where it differs from `docs/architecture_decisions.md`, the latter is what was built and measured.

Every statement below is tagged by evidence type:

- **[LIT]** — stated in a reference paper (see §6 for which one)
- **[EXP]** — measured by us on `mixalldata_clean.csv` (reproducible; see `docs/research_log.md`)
- **[DEC]** — a methodological decision (ours; can be revisited with evidence)
- **[INF]** — an inference/assumption, not directly verified

---

## 1. Objective

Develop an Explainable AI framework for vehicular misbehavior detection in C-ITS using the VeReMi dataset, integrating:

1. Machine-learning misbehavior detection
2. Explainable AI using SHAP
3. Large Language Models
4. Automated human-readable misbehavior reports

Final goal: a **reproducible end-to-end pipeline** that detects malicious vehicular messages and generates interpretable explanations suitable for researchers, transportation authorities and cybersecurity analysts.

### Design principles

| Component | Role | Must NOT |
|---|---|---|
| ML model | The detector. Sole source of the prediction. | — |
| SHAP | Explains why the ML model produced that prediction. | Be replaced by generic/global feature talk for a local report. |
| LLM | Converts a small structured evidence package into a readable report. | Detect, re-classify, override the prediction, or invent feature values, observations, causes or evidence. |
| Streamlit | Presentation layer over the same pipeline code. | Re-implement or diverge from the research pipeline. |

Reports must separate **observed/model evidence**, **model interpretation**, and **uncertainty**. The LLM layer must be provider-agnostic.

Pipeline: `VeReMi data → data understanding & quality → feature preparation → ML detection → prediction + confidence → SHAP → evidence package → LLM interpretation → human-readable report → reproducible framework/demo`.

---

## 2. Dataset: `data/mixalldata_clean.csv`

- 3,194,808 rows × 30 columns; 25 float64 + 5 int64; **no NaN, no inf** [EXP].
- Columns: `type, sendTime, sender, senderPseudo, messageID, class`, then for each of pos/spd/acl/hed: `x,y,z` and `x_n,y_n,z_n`.
- **Constant columns (drop):** `type` (always 4), and all eight z-columns `posz, posz_n, spdz, spdz_n, aclz, aclz_n, hedz, hedz_n` (always 0) [EXP].
- **Identifiers (never model features):** `messageID` (fully unique; corr. 0.987 with sendTime), `sender`, `senderPseudo` [EXP/DEC].
- **Usable per-message features (16):** `posx, posy, posx_n, posy_n, spdx, spdy, spdx_n, spdy_n, aclx, acly, aclx_n, acly_n, hedx, hedy, hedx_n, hedy_n`.
- `sendTime` spans 240.6 → 86,399.98 s, i.e. a **full 24-hour day**, with attackers present in every hour; traffic peaks ~07–09 h and ~17–19 h [EXP].
- Heading `(hedx, hedy)` is approximately a unit vector (median norm 1.0), but norms range 0–1.41, so it is not exactly normalised for all rows [EXP].
- `*_n` columns: `posx_n/posy_n` ≈ 4 (range 0–8.3), `spd*_n` signed and tiny (±0.1), `acl*_n` ≥ 0 and tiny, `hed*_n` ≥ 0 with mean ≈ 13 and max ≈ 121 [EXP]. Their exact semantics (noise sample vs. confidence range) — see §6 [LIT].

### 2.1 Provenance

- The file is the Kaggle dataset "veremi-extension-data-1-21-gb / mixalldata_clean.csv" (owner handle `ivarprudnikov`), as shown by the paths in two Kaggle notebook printouts that were consulted [LIT-K]. **Who built/cleaned it and how is not documented anywhere in the references.**
- Our sender counts — 7,399 misbehaving + 17,264 genuine = 24,663 (30.0 % attackers) — **match exactly** the "MixAll 0024" row of Table I in the VeReMi Extension paper (00h–24h, 23.29 veh/km², 30 % attacker penetration) [LIT-1 + EXP]. The CSV is therefore derived from the MixAll_0024 scenario.
- The paper reports 7.5 M + 11.95 M *received* messages for MixAll; our 3.19 M rows have no `rcvTime`/receiver, `type` is always 4 and `messageID` is unique ⇒ the CSV is built from the **sender-side ground-truth log** (one row per transmitted message), not from receiver logs [INF].
- Licence of the original dataset (Zenodo record, DOI 10.5281/zenodo.20090854): Creative Commons Attribution 4.0 International (CC BY 4.0), checked on 2026-09-21.
- Simulation stack: F2MD (a VEINS extension) on OMNeT++ + SUMO, LuST (Luxembourg) scenario, 1.61 km² sub-network [LIT-1].

---

## 3. Verified dataset findings [EXP]

All numbers below were re-measured during the dataset audit on the full file unless stated.

### 3.1 Class distribution (verified identical to the notebook)

| class | rows | % | senders | | class | rows | % | senders |
|---|---|---|---|---|---|---|---|---|
| 0 | 1,900,539 | 59.49 | 17,264 | | 10 | 43,264 | 1.35 | 390 |
| 1 | 43,653 | 1.37 | 390 | | 11 | 44,337 | 1.39 | 389 |
| 2 | 43,567 | 1.36 | 390 | | 12 | 43,118 | 1.35 | 390 |
| 3 | 43,857 | 1.37 | 390 | | 13 | 131,305 | 4.11 | 390 |
| 4 | 42,575 | 1.33 | 389 | | 14 | 126,724 | 3.97 | 389 |
| 5 | 41,925 | 1.31 | 390 | | 15 | 129,270 | 4.05 | 389 |
| 6 | 44,359 | 1.39 | 389 | | 16 | 175,391 | 5.49 | 389 |
| 7 | 42,258 | 1.32 | 389 | | 17 | 44,310 | 1.39 | 389 |
| 8 | 42,583 | 1.33 | 389 | | 18 | 86,883 | 2.72 | 390 |
| 9 | 42,790 | 1.34 | 389 | | 19 | 82,100 | 2.57 | 389 |

Key structural fact: **every misbehavior class has ≈ 389–390 senders**; the row-count differences between classes come entirely from *message rate*, not from the number of vehicles. Imbalance is ~30 % attacker vehicles vs 70 % genuine at sender level (7,399 vs 17,264).

### 3.2 Sender structure

- 24,663 senders; **each sender has exactly one class** (0 multi-class senders). Labels are per-vehicle, not per-message.
- The file is laid out as **24,663 contiguous sender blocks** (one run per sender), each block already sorted by `sendTime`. The file is not globally chronological (24,417 negative row-to-row time steps = block boundaries).
- Rows per sender: median 115, mean 129.5, 1 %: 6, 99 %: 611, min 1, max 1,494.
- `senderPseudo`: 118,909 unique. Classes 0–15: exactly 1 pseudonym per sender. Class 16: median 6 (max 7). Class 17: median 57 (max 100). Classes 18, 19: median 100.
- `senderPseudo == 1` occurs in 40,853 rows across 377 senders — **all in class 16**. It is a placeholder/artifact value shared by different vehicles, not a real identity.

### 3.3 Timing

- Within-sender gaps (properly computed, non-positive gaps included): n = 3,170,145; **zero gaps = 0, negative = 0, gaps > 1 s = 0**; max = 1.000 s. Sequences are complete — no missing beacons.
- Gap quantiles: 1 % 0.1667, 5 % 0.25, 25 %–100 % 1.0. 76.99 % of gaps are 1 s.
- **All sub-second gaps belong to classes 13–16, 18, 19** (99.5–99.8 % of their gaps < 1 s). Median gap: classes 13/14/15 = 0.25 s, class 16 = 0.167 s, classes 18/19 = 0.5 s. Every other class: exactly 1.0 s.

### 3.4 Duplicated feature vectors (new finding — important)

- **21.98 % of rows (702,131) have an 8-value kinematic vector `(posx,posy,spdx,spdy,aclx,acly,hedx,hedy)` that occurs more than once** in the dataset.
- 171,607 distinct vectors occur in **more than one class**. 128,142 class-0 rows are bit-identical to rows labelled as an attack.
- Share of each class's rows that are duplicates: class 10: 99.1 %, 11: 99.3 %, 13: 99.9 %, 15: 99.9 %, 17: 99.1 %, 19: 99.9 %, 9: 82.4 %, 12: 53.3 %; all others < 13 %.
- Consequence: for replay/disruptive-type classes, a **single message in isolation is literally indistinguishable from a genuine message** (it *is* a copied genuine message). Per-message features alone have a hard performance ceiling for these classes; separability must come from temporal/sender-context features (§5.2).

### 3.5 Per-class behavioural signatures (full data)

`gap` = within-sender inter-message time; `spd` = ‖(spdx,spdy)‖; `Δpos` = distance between consecutive positions of the same sender; `w/in-sender spd sd` = median over senders of the std of `spd`.

| cls | gap med | spd mean | spd==0 | w/in-sender spd sd | Δpos med | Δpos==0 | pos==(0,0) | dup-vector share | pseudonyms/sender (med) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.0 | 9.71 | 0.2 % | 4.19 | 11.9 | 0 % | 0 % | 6.7 % | 1 |
| 1 | 1.0 | 9.73 | 0.1 % | 4.25 | **0.0** | **99.1 %** | 0 % | 9.3 % | 1 |
| 2 | 1.0 | 9.82 | 0.1 % | 4.10 | 12.0 | 0 % | 0 % | 3.7 % | 1 |
| 3 | 1.0 | 9.55 | 0.2 % | 4.26 | **770** | 0 % | 0 % | 3.9 % | 1 |
| 4 | 1.0 | 9.65 | 0.1 % | 4.21 | **36.6** | 0 % | 0 % | 6.9 % | 1 |
| 5 | 1.0 | **31.1** | 0 % | **0.00** | 11.9 | 0 % | 0 % | 5.2 % | 1 |
| 6 | 1.0 | 13.6 | 0 % | 4.35 | 12.1 | 0 % | 0 % | 6.8 % | 1 |
| 7 | 1.0 | **30.6** | 0 % | **11.4** | 12.1 | 0 % | 0 % | 5.9 % | 1 |
| 8 | 1.0 | 13.7 | 0 % | 6.28 | 12.1 | 0 % | 0 % | 6.9 % | 1 |
| 9 | 1.0 | 1.84 | **80.9 %** | 3.41 | 0.0 | 80.7 % | 0 % | 82.4 % | 1 |
| 10 | 1.0 | 13.1 | 2.0 % | 8.78 | **336** | 2.6 % | 1.0 % | **99.1 %** | 1 |
| 11 | 1.0 | 10.5 | 1.1 % | 6.25 | 17.2 | 16.4 % | 0.1 % | **99.3 %** | 1 |
| 12 | 1.0 | 4.80 | 51.3 % | 5.49 | 0.0 | 50.4 % | **51.3 %** | 53.3 % | 1 |
| 13 | **0.25** | 9.73 | 0.1 % | 4.18 | 0.0 | 66.6 % | 0 % | **99.9 %** | 1 |
| 14 | **0.25** | **30.6** | 0 % | 11.4 | **769** | 0 % | 0 % | 1.7 % | 1 |
| 15 | **0.25** | 12.5 | 1.6 % | 7.72 | 289 | 3.6 % | 0.4 % | **99.9 %** | 1 |
| 16 | **0.167** | 2.99 | 0.9 % | 4.45 | 11.6 | 0.5 % | 0 % | 4.6 % | **6** |
| 17 | 1.0 | 10.7 | 0.8 % | 6.27 | 20.1 | 13.8 % | 0.1 % | **99.1 %** | **57** |
| 18 | **0.5** | **30.6** | 0 % | 11.4 | **765** | 0 % | 0 % | 13.0 % | **100** |
| 19 | **0.5** | 11.9 | 2.2 % | 7.36 | 271 | 3.1 % | 0.3 % | **99.9 %** | **100** |

Other observations:
- Class 12 senders begin with long runs of **all-zero** rows (position, speed, heading and noise all 0) before real values appear — 22,103 all-zero rows in class 12 (a few hundred in 10, 15, 19; tens in 11, 17).
- Class 9 senders drive normally and then freeze (speed 0, fixed position) — so the *early* messages of a class-9 sender are behaviourally genuine but carry label 9 (per-vehicle labelling ⇒ message-level label noise).

---

## 4. Class mapping — RESOLVED 2026-09-20: mapping F (§4.3) adopted

**Decision [DEC, author-directed]:** the original mapping had only been taken as an idea from Khan et al., and all decisions must be data-based, not assumed. Mapping **F** is therefore the project's class mapping. It is confirmed by three independent lines of evidence: (1) positive data tests for all 19 misbehavior classes (§4.5), (2) the F2MD simulator source code enum (§4.5), (3) two references (§4.4). The superseded mapping W and the evidence against it are kept below because the discrepancy with a published paper is itself a reportable finding.

### 4.1 Original working mapping W (from Khan et al. 2025, Table 2) — SUPERSEDED, do not use

`0 Normal, 1 Constant position, 2 Constant position offset, 3 Random position, 4 Random position offset, 5 DoS, 6 DoS random, 7 DoS random sybil, 8 Data replay, 9 Data replay sybil, 10 Disruptive, 11 DoS disruptive sybil, 12 DoS disruptive, 13 Eventual stop, 14 Grid sybil, 15 Constant speed, 16 Constant speed offset, 17 Delayed messages, 18 Random speed, 19 Random speed offset`

### 4.2 What the data says [EXP]

The working mapping is **contradicted by measured behaviour for 14 of 20 classes** (it is consistent only for 0–4 and 10):

| cls | Working name | Measured behaviour (§3.5) | Consistent? |
|---|---|---|---|
| 5 | DoS | 1.0 s gaps (no flooding); speed perfectly constant within each sender (sd 0.00) | ✗ — behaves as *constant speed* |
| 6 | DoS random | 1.0 s gaps; speed shifted up by a fixed amount | ✗ — behaves as *constant speed offset* |
| 7 | DoS random sybil | 1.0 s gaps; 1 pseudonym; speed uniform-random (mean 30.6, sd 11.4) | ✗ — behaves as *random speed* |
| 8 | Data replay | 93 % unique vectors (nothing replayed); noisy speed | ✗ — behaves as *random speed offset* |
| 9 | Data replay sybil | 1 pseudonym; 81 % of messages at speed 0, frozen position | ✗ — behaves as *eventual stop* |
| 11 | DoS disruptive sybil | 1.0 s gaps; 1 pseudonym; 99.3 % replayed vectors | ✗ — behaves as *data replay* |
| 12 | DoS disruptive | 1.0 s gaps; leading all-zero rows | ✗ — behaves as *delayed messages* |
| 13 | Eventual stop | 0.25 s gaps (4× message rate); speed never 0 | ✗ — behaves as *DoS* |
| 14 | Grid sybil | 0.25 s gaps; 1 pseudonym; random position+speed | ✗ — behaves as *DoS random* |
| 15 | Constant speed | 0.25 s gaps; speed sd 7.7 (not constant); 99.9 % replayed | ✗ — behaves as *DoS disruptive* |
| 16 | Constant speed offset | 0.167 s gaps; ~6 pseudonyms per sender | ✗ — behaves as *grid sybil* |
| 17 | Delayed messages | ~57 pseudonyms per sender; 99.1 % replayed | ✗ — behaves as *data replay sybil* |
| 18 | Random speed | 0.5 s gaps; 100 pseudonyms; random pos+speed | ✗ — behaves as *DoS random sybil* |
| 19 | Random speed offset | 0.5 s gaps; 100 pseudonyms; 99.9 % replayed | ✗ — behaves as *DoS disruptive sybil* |

### 4.3 ADOPTED mapping F [EXP + simulator source + LIT]

`0 Genuine, 1 Constant position, 2 Constant position offset, 3 Random position, 4 Random position offset, 5 Constant speed, 6 Constant speed offset, 7 Random speed, 8 Random speed offset, 9 Eventual stop, 10 Disruptive, 11 Data replay, 12 Delayed messages, 13 DoS, 14 DoS random, 15 DoS disruptive, 16 Grid sybil, 17 Data replay sybil, 18 DoS random sybil, 19 DoS disruptive sybil`

### 4.4 Where the three circulating mappings come from [LIT]

| Mapping | Source | Status vs. our data |
|---|---|---|
| **W** — the working mapping (§4.1) | Khan et al. 2025 (Computer Networks 270, 111575), Table 2 — identical names *and* identical row counts to ours | Contradicted for 14 classes (§4.2). Note Khan's own confusion-matrix indices don't follow their Table 2 either. |
| **K** — Kaggle notebooks | Alladi et al. 2023 (DCAN 9), Table 1 — an 18-class scheme that *drops* Eventual stop and Delayed messages; notebooks mark 18/19 "Unknown"/"Extended Attack A/B" | Agrees with data only for 0–8; wrong for 9–17 |
| **F** — candidate (§4.3) | VeMisNet (Youness et al. 2025, JSAN 14:100) Table 3 lists the 20 classes in exactly this order; MistralBSM (arXiv:2407.18462) Table I gives codes A0 Genuine, A1 ConstPos, A5 ConstSpeed, A9 EventualStop, A11 DataReplay, A12 DelayedMessages, A13 DoS, A14 DoSRandom, A18 DoSRandomSybil — all consistent with F | **Consistent with every measured signature** |

- The VeReMi Extension paper itself assigns **no numeric IDs** — it only names and defines the 19 misbehaviors. The numbering comes from the F2MD simulator that generated the dataset (verified, §4.5).
- Naming: the paper calls class 16 "Traffic congestion Sybil" (a.k.a. Grid Sybil) and class 12 "Delayed Messages" (VeMisNet: "Stale Messages").
- **Fault vs attack grouping per the VeReMi Extension paper:** malfunctions (faults) = the 8 position/speed variants **+ Delayed Messages**; attacks = DoS, DoS Random, DoS Disruptive, Data Replay, Disruptive, Eventual Stop, Traffic-congestion (Grid) Sybil, and the three Sybil variants. Under F: faults = {1–8, 12}, attacks = {9, 10, 11, 13–19}.
- Consequence for literature use: **SHAP/class-level interpretations in Khan et al. rest on wrong class names** and must not be cited as class-specific evidence.

### 4.5 Verification of mapping F (2026-09-20) — `scripts/class_mapping_verification.py` + F2MD source

**Simulator source [external, verified by fetch]:** `veins-f2md/src/veins/modules/application/f2md/mdEnumTypes/AttackTypes.h` defines `enum Attacks { Genuine=0, ConstPos, ConstPosOffset, RandomPos, RandomPosOffset, ConstSpeed, ConstSpeedOffset, RandomSpeed, RandomSpeedOffset, EventualStop, Disruptive, DataReplay, StaleMessages, DoS, DoSRandom, DoSDisruptive, GridSybil, DataReplaySybil, DoSRandomSybil, DoSDisruptiveSybil, MAStress }` — i.e. exactly F for 0–19 (`StaleMessages` = the paper's "Delayed Messages"; `MAStress`=20 is not in our data).

**F2MD default parameters (`F2MDParameters.h`) vs. our measurements** — note these are repository defaults, not a documented record of the dataset run; the agreement with the data is what matters:

| F2MD default | Our data [EXP] |
|---|---|
| `DosMultipleFreq = 4` | classes 13–15: gap 0.25 s = 4× the 1 s beacon rate |
| `DosMultipleFreqSybil = 2` | classes 18–19: gap 0.5 s |
| `SybilVehNumber = 5` (+ the attacker itself = 6 identities) | class 16: median 6 pseudonyms/sender, gap 1/6 s = 0.167 s |
| `StaleMessages_Buffer = 60` | class 12: **every** sender starts with an all-zero run, median length 61 messages (= 61 s); zero rows occur nowhere else |
| `RandomSpeedX/Y = 40` | classes 7, 14, 18: speed components uniform on 0…40, lag-1 autocorrelation ≈ 0 |
| `RandomAccelX/Y = 2` | classes 14, 18: mean acceleration component ≈ 1.0 |

**Positive data tests per class [EXP]** (v_true := Δposition/Δt of the same sender; "off-road" := 5 m cell never visited by any genuine vehicle; held-out genuine baseline off-road = 0.01 %):

| cls | Name (F) | Defining property per LIT-1 | Measured |
|---|---|---|---|
| 1 | Constant position | position frozen | 99.1 % zero displacement; 91 % off-road |
| 2 | Constant position offset | true pos + fixed Δ | kinematics identical to genuine (speed-vs-Δpos residual sd 0.676 vs 0.677) **but 86 % of rows off-road** |
| 3 | Random position | uniform over playground | posx uniform on 0…1500 (sd 434 vs 433 theoretical); lag-1 autocorr 0.002; speed untouched (autocorr 0.97) |
| 4 | Random position offset | true pos + random Δ | Δpos inconsistent with speed (residual sd 18.4); 66 % off-road; median jump 37 m |
| 5 | Constant speed | speed fixed | within-sender speed sd = 0.000; positions on-road (genuine) |
| 6 | Constant speed offset | true speed + fixed Δ | per-component error vs v_true: **mean ≈ 7, sd 0.73 (= genuine baseline 0.72)** → constant bias |
| 7 | Random speed | uniform random speed | components uniform 0…40, autocorr ≈ 0; positions genuine (autocorr 0.99) |
| 8 | Random speed offset | true speed + random Δ | per-component error: **mean ≈ 0, sd 7.4** → zero-mean random perturbation |
| 9 | Eventual stop | freeze position, speed → 0 | 80.9 % speed = 0; 79.9 % of messages identical to previous; normal driving before |
| 10 | Disruptive | replay from **random** neighbours | 69 % of messages are exact copies of genuine vehicles' messages; source changes between consecutive copies 90 % of the time |
| 11 | Data replay | replay from a **target** neighbour | 86 % exact copies; source switch rate only 43 %; ~14 targets over a trip |
| 12 | Delayed (stale) messages | correct data, sent late | kinematically genuine and on-road after a ~60-message zero buffer (matches `StaleMessages_Buffer`) |
| 13 | DoS | rate above limit | 0.25 s gaps; 66.6 % of messages repeat own previous state; content genuine |
| 14 | DoS random | DoS + all fields random | 0.25 s; uniform random pos & speed; 1 pseudonym |
| 15 | DoS disruptive | DoS + disruptive | 0.25 s; 72 % copies; source switch 89 % |
| 16 | Grid (traffic-congestion) sybil | grid of ghost vehicles | ~6 pseudonyms; 0.167 s; pseudonym changes on 90 % of consecutive messages |
| 17 | Data replay sybil | new identity per new target | 83 % copies; source switch 42 % (same as class 11); **P(pseudonym change \| target change) = 99.2 %** vs 13.9 % otherwise |
| 18 | DoS random sybil | DoS random + identity change per message | 0.5 s; random fields; pseudonym changes on 99.6 % of messages |
| 19 | DoS disruptive sybil | DoS disruptive + identity change | 0.5 s; 72 % copies; source switch 89 %; pseudonym changes on 98.7 % |

Honest residual: class 12's *delay* itself cannot be measured (the CSV has no true trajectory to compare against); its identity rests on the zero-buffer signature matching the simulator's buffer size, genuine-looking kinematics, the simulator enum, and elimination.

**Rule:** the mapping lives in exactly one place in code (a config file). Class definitions shown to the LLM/user are taken from LIT-1 wording (§6.2). A wrong mapping would make every LLM report describe the wrong attack — it was the single highest-impact correctness issue found in the initial audit.

---

## 5. Methodological foundations

### 5.1 Evaluation split [DEC, driven by EXP §3.2]

- Labels are per-sender and sender blocks are contiguous ⇒ a random row split puts near-identical consecutive messages of the same vehicle in train and test. **Primary evaluation = sender-grouped, class-stratified split** (e.g. `StratifiedGroupKFold` / a fixed stratified group hold-out on `sender`). With ≥ 389 senders in every class this is comfortably feasible (e.g. 70/15/15 by sender).
- Also report a **random row split as a secondary "leakage-inflated" comparison**, so the size of the leakage effect is itself a documented finding, and so our numbers can be compared with literature that uses row splits (§6).
- Split file (sender → fold) is generated once with a fixed seed and saved; every experiment reuses it.
- Any fitted preprocessing (scalers, encoders, SHAP background) is fit on train senders only.

### 5.2 Feature strategy [DEC, driven by EXP §3.4–3.5]

Per-message features cannot separate replay/DoS/sybil classes from genuine traffic (§3.4). Planned, in increasing richness, evaluated as an ablation:

- **F0 (per-message baseline):** the 16 raw features (+ cheap derived: speed magnitude, acceleration magnitude, heading angle).
- **F1 (sender-context, causal):** computed per identity from the *current and previous* messages only — inter-message gap, Δposition, Δposition vs. speed×gap consistency, Δspeed vs. acceleration consistency, Δheading, short rolling statistics (e.g. last 5–10 messages), message rate.
- Features must be **causal** (no look-ahead) so that the demo can score a message as it arrives.
- **Open decision — which identity to group by:** `sender` is simulator ground truth; a real receiver only sees `senderPseudo`. Grouping by `sender` makes sybil classes trivially visible (pseudonym count) and is arguably unrealistic; grouping by `senderPseudo` is realistic but fragments sybil senders (classes 17–19 have 1–2 messages per pseudonym; per-pseudonym gap for 18/19 is 50 s). Note `senderPseudo == 1` is a shared placeholder in class 16 and must not be treated as one identity. Evaluate both; state clearly which one the headline results use.
- Raw `posx/posy` encode map location. Keep them (constant/random-position attacks need them) but watch for the model memorising locations; Δ-features are the safer signal.
- `sendTime` (absolute) is not a feature; only differences are.
- Tree models need no scaling; scaling is applied inside the pipeline only for linear/MLP models.

### 5.3 Known label-noise sources (report as limitations, do not "fix" silently)

- Per-vehicle labels: early messages of class-9 senders are behaviourally genuine; zero-filled warm-up rows in class 12.
- Replay-type attack rows are exact copies of genuine rows (different class, identical features).
- Hence per-class recall ceilings < 100 % are expected and legitimate at message level. Consider reporting **sender-level (vehicle-level) detection** as a complementary metric (aggregate message predictions per sender).

### 5.4 Models [DEC — provisional]

Sensible comparison set for 3.2 M rows × ~16–35 features on a CPU laptop with SHAP compatibility:
Logistic Regression (linear baseline) → Decision Tree (interpretable baseline) → Random Forest → **XGBoost `hist`** and/or **HistGradientBoosting** (LightGBM only if installed without friction). MLP optional, only if it adds something. Final model chosen on grouped-validation macro-F1, with TreeSHAP compatibility and inference cost as tie-breakers. Hyper-parameter search kept small and on a sender-subsample.

### 5.5 Metrics

Macro-F1 (primary), weighted-F1, balanced accuracy, accuracy (reported but never alone), per-class precision/recall/F1, confusion matrix (row-normalised), training time, inference latency per message, model size. Secondary semantic grouping (genuine / fault-like / attack; or binary genuine-vs-misbehaving) reported from the same 20-class predictions, not from a separately trained model, unless justified.

### 5.6 Imbalance

Class 0 = 59.5 % of rows. Start with class weights / no resampling; judge by macro-F1 and per-class recall. No SMOTE by default (synthetic kinematic points are hard to justify, and the large classes are large only because of message rate).

### 5.7 SHAP [DEC]

- `TreeExplainer` on the final tree model (exact, fast). Multiclass ⇒ SHAP values per class; for a local report use the SHAP vector of the **predicted class** (and optionally the runner-up class for uncertainty).
- State the output space explicitly (raw margin/log-odds vs. probability) in every figure and in the evidence package — SHAP values on margins are not probability points.
- Global: mean |SHAP| per feature overall and per class, on a stratified sample of **test senders**. Local: top-k contributions with actual feature values.
- Faithfulness checks: additivity (base value + Σ SHAP = model output), and a deletion/perturbation sanity check on a sample.

### 5.8 LLM reporting [DEC]

- Input = one JSON **evidence package**: message identifiers, predicted class id + name + definition, confidence and top-k class probabilities, actual feature values (with units), top-k SHAP features with signed values and direction, base value/output space, model name/version, known limitations. Nothing else.
- Provider-agnostic interface (`generate(evidence) -> report`) with at least: a deterministic **template backend** (no LLM; always works, offline, used for tests and as a faithfulness baseline) and one or more LLM backends selected by config/env var. No API keys in code or notebook.
- Prompt contract: use only supplied evidence; never change the prediction; mark uncertainty when confidence is low or top-2 classes are close; fixed report sections — *Summary · Observed evidence · Model interpretation · Uncertainty & limitations · Suggested analyst action*.
- **Automatic faithfulness validation** of every report: every number quoted must exist in the evidence package; predicted class in text must equal the ML prediction; every feature named must be in the package. Report the pass rate as a Phase 8 result.

#### 5.8.1 LLM provider strategy (2026-09-20) [DEC: API preferred, modular, decision NOT locked]

**What the LLM layer actually needs [INF from design]:** one call per report; ≈ 1–1.5 k input tokens (prompt contract + evidence JSON) and ≈ 300–500 output tokens ⇒ ~2 k tokens/report. Volumes: demo = a handful per session; Phase 8 faithfulness evaluation = ~300–500 reports per model (≈ 1 M tokens). Needs solid instruction-following at temperature 0; no fine-tuning, no long context, no tools. The data is simulated (no personal data), so free tiers that train on prompts are acceptable, but note it in the write-up.

**Hardware [EXP]:** i5-12500H (16 threads), 15.7 GB host RAM, RTX 3050 Laptop **4 GB VRAM**; Ollama not installed. ⇒ local models up to ~3–4 B parameters (4-bit) fit fully on the GPU; 7–8 B only with partial CPU offload (slow). Local is viable as an **offline fallback**, not as the headline model — consistent with the preference for an API-based writer.

**Free-tier landscape (secondary sources, mid-2026; official pages show limits only inside each account's console — numbers conflict between sources and change often, so NOTHING in the design may depend on them; re-check in the console after sign-up):**

| Option | Reported free limits | Card | Trains on prompts | OpenAI-compatible | Fit |
|---|---|---|---|---|---|
| **Groq** | ~30 RPM; ~1 k req/day; 100–200 k tokens/day on 70B/120B-class open models, ~500 k/day on 8B | No | No | Yes | **First candidate**: ≈ 50–100 reports/day on a large model, enough for dev + evaluation spread over a few days |
| **Google AI Studio (Gemini)** | Flash-Lite ~500–1,000 req/day; full Flash reported as low as ~20/day after 2025–26 cuts | No | Yes (outside EU/UK) | Partial (compat endpoint) | Second candidate; good daily volume on Flash-Lite |
| **OpenRouter** | ~50 req/day on `:free` models (≈ 1,000/day after a one-time $10 top-up); 20 RPM | No | Depends on routed provider | Yes | Useful single key to compare several models |
| **Mistral (Experiment tier)** | very large monthly token quota, low RPM | No | Yes (opt-in required) | Yes | Backup |
| Cerebras / GitHub Models | sources report free tier withdrawn / service retired in July 2026 | — | — | — | Do not plan on these |
| **Ollama (local)** | unlimited, offline | — | No | Yes (`/v1` endpoint) | Offline fallback with a 3–4 B model |
| Paid commercial APIs | — | Yes | — | via same interface | Drop-in later if a key/budget appears |

**Architecture that keeps the decision reversible:**
1. One interface: `Reporter.generate(evidence: dict) -> Report`. Backends selected purely by config.
2. `template` backend (no LLM, deterministic) — always available; used in tests, as the faithfulness baseline, and as automatic fallback when no key/network/quota.
3. **One `openai_compatible` backend covers Groq, OpenRouter, Mistral, Gemini's compatibility endpoint AND Ollama** — switching provider = changing `base_url`, `model`, `api_key_env` in `config.yaml`. Native SDK backends are added only if a provider-specific feature is ever needed.
4. API keys only via environment / `.env` (git-ignored). Never in code, notebooks or config.
5. **Disk cache** of reports keyed by hash(evidence + prompt version + provider + model): makes results reproducible, makes re-runs free, protects against rate limits, and lets the Streamlit demo work offline from cached reports.
6. Every report stores provenance: provider, model id, prompt version, timestamp, validator result.
7. Temperature 0; retry with back-off on rate-limit errors; hard daily budget guard.
8. Research bonus: because backends are interchangeable, Phase 8 can report **faithfulness rate by backend** (template vs. small vs. large model) — a result none of the references has.

**Outcome (2026-09-21):** the Groq free tier was used with `openai/gpt-oss-20b` and `openai/gpt-oss-120b` (the Llama 3.3 70B model named by secondary sources no longer existed there); see AD-11, AD-14 and AD-15 in `docs/architecture_decisions.md` and the evaluation in `docs/research_log.md`.

### 5.9 Engineering rules

- Code in a small package (`src/`), thin notebooks that import from it, one config file (paths, seed, class mapping, feature lists), relative paths only, fixed seeds, saved artifacts (split, model, metrics JSON, SHAP samples), `requirements.txt` with pinned versions, README with run order.
- Never overwrite existing results in `outputs/` silently — new runs go to new, named files/folders.
- Only claim numbers that were actually measured and saved.
- The CSV, `.venv`, `.env` and the large intermediate files are git-ignored.

---

## 6. Literature foundations

All 11 unique papers listed below were read in full on 2026-09-20. The PDFs are copyrighted and are not part of this repository; obtain them through the citations. None of the statements below may be strengthened beyond what is written here without re-checking the paper.

### 6.1 Reference index

| Tag | Reference | Use for us |
|---|---|---|
| LIT-1 | Kamel, Wolf, van der Heijden, Kaiser, Urien, Kargl — *VeReMi Extension: A Dataset for Comparable Evaluation of Misbehavior Detection in VANETs*, IEEE ICC 2020 | **Dataset paper**: definitions, generation, sensor-error model, baseline |
| LIT-2 | Khan et al. — *Transformer-based explainable AI approach using SHAP for intrusion detection in VANETs*, Computer Networks 270 (2025) 111575 | Closest prior work (same CSV, 20-class, SHAP). Source of working mapping W. Cautionary example |
| LIT-3 | Alladi, Kohli, Chamola, Yu — *Deep learning based misbehavior classification scheme for IDS in C-ITS*, DCAN 9 (2023) 1113–1122 | Sequence DL, 18 classes; confusable-pair analysis; source of mapping K |
| LIT-4 | Hamhoum & Cherkaoui — *MistralBSM*, arXiv:2407.18462v1 (2024) | **LLM-as-detector** contrast paper; A-codes support mapping F |
| LIT-5 | Alan et al. — *Enhancing Anomaly Detection in Autonomous Systems with Explainable AI*, J. Phys.: Conf. Ser. 3191 (2026) 012013 | XGBoost + SHAP, binary; cautionary example (AUC 0.66; prose contradicts figures) |
| LIT-6 | Youness et al. — *VeMisNet*, J. Sens. Actuator Netw. 14 (2025) 100 | Best feature-set ablation + per-class tables; Table 3 supports mapping F |
| LIT-7 | Slama, Alaya, Zidi — *Guided ML misbehavior detection*, Inteligencia Artificial 25(70) (2022) 138–154 | **Same 3.19 M-row CSV, per-message classical ML** — our most direct baseline |
| LIT-8 | Mahesh et al. — *HyperLSTM*, Int. J. Distrib. Sens. Netw. 2025, 6789771 | Same Kaggle CSV, binary, per-sender windows |
| LIT-9 | Naqvi, Chaudhary, Kumar — *Neuro-Genetic framework*, IJACSA 15(2) 2024 | GA+ANN; 13 unnamed classes, ~30k rows; not comparable |
| LIT-10 | Karhana, Saini, Jaekel — *False alerts in safety messages*, Network 6 (2026) 53 | Different dataset (VeReMiAP); derived-feature ideas |
| LIT-K | Two Kaggle notebook printouts (LSTM binary; async federated learning, never executed) | Provenance of the CSV only; not citable science |

### 6.2 What the dataset paper states [LIT-1]

- 19 misbehaviors = **9 malfunctions** (constant / random / constant-offset / random-offset for position and for speed, + delayed messages) and **10 attacks** (DoS, DoS Random, DoS Random Sybil, Data Replay, Data Replay Sybil, Disruptive, DoS Disruptive, DoS Disruptive Sybil, Eventual Stop, Traffic-congestion Sybil). Malfunction = non-malicious faulty OBU/sensor; attack = intentional.
- Definitions (paraphrased closely): *constant position* — position fixed for the whole simulation; *random position* — uniform over the playground; *offsets* — true value + fixed Δ / + uniform random Δ; speed malfunctions analogous on Vx, Vy; *delayed messages* — correct data sent with delay Δt; *DoS* — sending frequency above the standard's limit; *DoS Random* — DoS with all fields random; *Data Replay* — replays data previously received from one **target** neighbour (Sybil mode: new identity per target); *Disruptive* — replays data from **random** neighbours (also in DoS and Sybil modes); *Eventual Stop* — freezes position and sets speed to zero; *Traffic-congestion Sybil* — a grid of ghost vehicles, each with its own identity and a correct per-identity message frequency.
- **Not stated in the paper:** any numeric class IDs; offset sizes; random ranges; DoS rate multiplier; delay value; pseudonym policy; beacon interval; semantics of the `*_noise` fields (only a range "R[0,+∞]" — which our data contradicts, since `spd*_n` is signed). ⇒ **Do not let the LLM assert a physical meaning for `*_n` columns** beyond "noise/uncertainty value reported with the field".
- Sensor error model: position error ~ smoothed U(−5,5) m; speed error proportional to speed; heading error U(−20,20)·e^(−0.1·v) (shrinks with speed — explains large `hed*_n` at low speed) [LIT-1, p.3].
- The paper says attackers are labelled as such *only while actively attacking*. **Our CSV does not behave that way** — labels are strictly per-vehicle (§3.2) [EXP]; LIT-3 explains such labels come from log file names.
- Baseline in the paper: threshold plausibility/consistency checks, **binary**, per message: MixAll F1 = 0.899 (P 0.991, R 0.823); hardest = DataReplaySybil (F1 0.51), ConstSpeedOffset, DoSDisruptiveSybil, EventualStop, TrafficSybil. Paper warns accuracy is unsuitable (imbalance) and recommends F1. No multiclass results.
- Our measured message rates [EXP] (1 s normal; 0.25 s DoS; 0.5 s DoS-sybil; 0.167 s grid-sybil per physical sender) are data facts, not from the paper.

### 6.3 Sanity-check levels from the literature (use to judge our own numbers)

| Setting | Reported | Source | Caveat |
|---|---|---|---|
| 20-class, per-message, raw kinematics (+noise), IDs dropped, RF | **P/R/F1 ≈ 0.71/0.69/0.70** | LIT-7 (same CSV) | split unit unstated (70/30) |
| 20-class, sequences, basic kinematics | acc 0.72–0.81, F1 0.64–0.77 | LIT-6 | random sequence split |
| 20-class, + temporal/communication features, no IDs | acc 0.918, F1 0.909, **balanced acc 0.739** | LIT-6 | random sequence split |
| 20-class with IDs / timestamps as inputs | 96–99.8 % | LIT-2, LIT-6 | identifier leakage; LIT-2 also SMOTE-before-split (inferred) |
| Binary, per-message, raw kinematics, RF | F1 ≈ 0.94 | LIT-7 | — |
| Binary XGBoost, no IDs, receiver logs | acc 0.61, AUC 0.66 | LIT-5 | weak setup |

Rule of thumb [DEC]: **any 20-class score above ~0.95 in our experiments must be audited for identifier/time leakage or sender overlap before it is reported.** Classes the literature consistently finds hard: position/speed *offset* faults (confused with genuine), Data Replay ↔ Disruptive, Eventual Stop ↔ Genuine, Data Replay Sybil, and DoS unless rate features exist — matching our §3.4–3.5 findings.

### 6.4 Weaknesses in the literature that our methodology must avoid

1. **No reference uses a sender-disjoint split**; none discusses sender leakage (LIT-6 discusses leakage only for SMOTE). Several don't state the split at all (LIT-3, LIT-4).
2. **Identifiers/time as features**: in LIT-2 the top SHAP feature is `messageID`, top tree-importance features are `senderPseudo`/`sender`; in LIT-5 the top feature is `rcvTime`. These are artefacts, not explanations.
3. Balancing before splitting (LIT-2, inferred from test-set sizes); overlapping windows across random splits (LIT-3, LIT-8, LIT-K); test set reused for early stopping (LIT-5, LIT-K).
4. Shallow XAI: explainer type unstated; 20-class model explained as if binary; one local example; no faithfulness check; **explanatory prose contradicting the computed figures (LIT-5)** — exactly the failure our LLM layer must be engineered against.
5. Four incompatible class numberings across the references (§4.4).

### 6.5 Positioning (claim only "among the surveyed references")

- None of the 11 references generates a human-readable report from prediction + SHAP evidence. The only LLM paper (LIT-4) uses the LLM **as the classifier** (linear head, no text output); an LLM "report" is mentioned there only conceptually.
- None evaluates explanation faithfulness (SHAP→model or text→SHAP), and none gives per-class local explanations for the 20-class problem.
- Hence our defensible contributions: (a) leakage-aware (sender-disjoint) 20-class evaluation with the leakage effect quantified; (b) evidence-based feature design grounded in the duplicate-vector finding; (c) per-prediction SHAP evidence packages; (d) an LLM reporting layer with an explicit evidence contract and an **automatically measured faithfulness rate**; (e) a corrected, data-verified class mapping for this widely used CSV.
- Useful practices to adopt: per-class P/R/F1 + macro-F1 + balanced accuracy + MCC (LIT-2, LIT-6); bootstrap CIs and calibration/ECE (LIT-6); class weights rather than SMOTE (LIT-5, LIT-10); feature-set ablation with a significance test (LIT-6, LIT-10); inference-time reporting (all); derived magnitude/direction-consistency features (LIT-10), inter-message time and message rate (LIT-6); fault-vs-attack vocabulary (LIT-1, LIT-7).
