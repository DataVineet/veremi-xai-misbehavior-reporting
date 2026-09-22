# Research log

Dated, append-only log of what was measured or decided. Newest entry at the bottom. Only record things that were actually run; cite the script/notebook cell that produced each number.

---

## 2026-08-30 → 2026-09-06 — Initial data analysis (author, `VeReMi_XAI_Framework.ipynb`)

- Phases 1–3 (partial): row/class counts, missing values, exact-duplicate rows (none), sender→class consistency, pseudonym audit, global and within-sender timing, constant/unique columns, overall and class-wise feature statistics, noise magnitudes, class-wise boxplots.
- Saved: 7 figures in `outputs/figures/`, 2 CSVs in `outputs/results/`.

## 2026-09-20 — Dataset audit (`scripts/dataset_audit.py`, read-only, full dataset)

**Re-verified (identical to notebook):** 3,194,808 rows × 30 cols; 20 classes 0–19 with the same counts; no NaN/inf; `messageID` unique; 24,663 senders, each with exactly one class; 118,909 pseudonyms; constant columns = `type` + all eight z-columns.

**Corrected:** the notebook's "no zero/negative within-sender gaps" check was vacuous (it filtered `gap > 0` before counting). Proper check: 3,170,145 gaps, 0 zero, 0 negative, 0 above 1 s. Conclusion stands.

**New findings:**
1. File layout = 24,663 contiguous, time-sorted sender blocks.
2. Every misbehavior class has 389–390 senders; row-count imbalance among attack classes is purely message rate.
3. Sub-second gaps exist only in classes 13, 14, 15 (0.25 s), 16 (0.167 s), 18, 19 (0.5 s).
4. Multiple pseudonyms exist only in classes 16 (~6), 17 (~57), 18, 19 (~100). `senderPseudo == 1` is a shared placeholder: 40,853 rows, 377 senders, all class 16.
5. 21.98 % of rows share their 8-value kinematic vector with another row; 171,607 vectors occur in more than one class; 128,142 class-0 rows are bit-identical to attack-labelled rows. Classes 10, 11, 13, 15, 17, 19 are > 99 % duplicated vectors.
6. Class 5: within-sender speed sd = 0.00. Class 9: 80.9 % of messages at speed 0. Class 12: 51.3 % all-zero rows at the start of sender sequences. Class 1: 99.1 % zero displacement. Classes 3, 14, 18: median jump ≈ 770 m between consecutive messages.
7. `sendTime` covers a full 24 h; attackers present in all 24 hours.
8. **The working class-name mapping is inconsistent with measured behaviour for classes 5–9 and 11–19** (details: `docs/background.md` §4). Mapping NOT changed at this stage.

**Notebook defects recorded:** `behavioral_features` never defined; hard-coded absolute path; meaningless heading noise-to-signal ratio (unit mismatch); out-of-order execution.

**Not done (by instruction):** no feature engineering, no model training, no SHAP, no LLM work.

## 2026-09-20 — Reference reading (all 11 unique PDFs, full text)

- CSV provenance: Kaggle "veremi-extension-data-1-21-gb" (owner `ivarprudnikov`); sender counts 7,399 / 17,264 match VeReMi Extension Table I "MixAll 0024" exactly. Cleaning procedure undocumented.
- The VeReMi Extension paper assigns **no numeric class IDs**. The working mapping is Khan et al. 2025 Table 2 (same row counts as our CSV). Kaggle notebooks use Alladi et al.'s 18-class table (wrong for 9–17 on this CSV). VeMisNet Table 3 and MistralBSM A-codes give the numbering that matches our measured signatures.
- No reference uses a sender-disjoint split; several feed `messageID`/`sender`/`senderPseudo`/timestamps to the model (Khan et al.: `messageID` is the top SHAP feature).
- Most comparable baseline: Slama et al. 2022, same CSV, per-message RF, 20-class F1 ≈ 0.70.
- No reference generates text reports from SHAP evidence; MistralBSM uses the LLM as classifier only.
- Details and citations: `docs/background.md` §4.4 and §6.

## 2026-09-20 — Class mapping verified and adopted (`scripts/class_mapping_verification.py`)

- Decision rule: decisions must be data-based; the Khan et al. mapping had only been taken as an idea.
- F2MD simulator source (`veins-f2md/.../mdEnumTypes/AttackTypes.h`) fetched: enum order = mapping F exactly (0 Genuine … 19 DoSDisruptiveSybil; `StaleMessages` = Delayed messages).
- F2MD default parameters match measurements: DoS ×4 → 0.25 s; DoS-sybil ×2 → 0.5 s; 5 sybil ghosts + self → 6 pseudonyms / 0.167 s; stale buffer 60 → class-12 leading zero-run median 61; random speed max 40; random accel max 2.
- New positive tests: class 2 kinematically genuine but 86 % off-road (const. position offset); class 6 velocity error mean ≈ 7 / sd 0.73 (constant bias) vs class 8 mean ≈ 0 / sd 7.4 (random); class 11 & 17 replay one target at a time (source switch ≈ 43 %) vs classes 10, 15, 19 random neighbours (≈ 89–90 %); class 17 changes pseudonym on 99.2 % of target changes.
- **Mapping F adopted** (docs/background.md §4). Khan et al. Table 2 documented as inconsistent with the dataset.

## 2026-09-20 — LLM provider reconnaissance (no implementation)

- Constraints: no API key yet; API-based LLM preferred (limited local GPU); integration must stay modular/switchable incl. Ollama; decision not to be locked.
- Hardware measured: i5-12500H, 15.7 GB RAM, RTX 3050 Laptop 4 GB VRAM; Ollama not installed.
- Free-tier survey (secondary sources, mid-2026; official Groq/Gemini pages expose limits only in the account console): Groq and Google AI Studio are the practical no-card options; OpenRouter as aggregator; Cerebras free tier / GitHub Models reported withdrawn. Numbers are unverified until checked in our own console.
- Design consequence: a single OpenAI-compatible backend + template backend + disk cache makes the provider a config line (docs/background.md §5.8.1).

## 2026-09-20 — Implementation of Phases 4–8 

All numbers: test split = 3,710 vehicles / 478,186 messages never seen in training (sender-disjoint 70/15/15, seed 42). Source: `outputs/results/experiments/*.json`.

1. **F0 per-message features, XGBoost:** macro-F1 0.369 (acc 0.734). Same with a random row split: 0.680 → the ≈ 0.70 of Slama et al. on this CSV is reproduced only under leakage.
2. **F1 = F0 + 22 causal history features keyed on pseudonym:** macro-F1 0.908. Keyed on the true sender: 0.846 (worse — hides pseudonym switching).
3. **Error analysis of F1:** residual errors are mostly structural — Grid Sybil→Genuine 16.2 % (≈ 1/6 = the attacker's own beacons), Eventual stop→Genuine 18.5 % (pre-stop driving), Delayed→Genuine 31 %, plus one fixable: Constant position offset→Genuine 41 %.
4. **Road map from genuine training traffic → `road_dist` (F2):** genuine test rows > 5 m off-map 0.00 %, class 2: 80 %. Class-2 F1 0.738 → 0.929; macro-F1 0.919.
5. **Models on F2 (test macro-F1):** LogReg 0.800, Decision tree 0.907, Random forest 0.912, XGBoost 0.919; class-weighted XGBoost 0.908 (bal. acc 0.904). Small hyper-parameter check: val macro-F1 0.9142–0.9158 → plateau. Final: `xgb_F2_pseudo_d6`, test macro-F1 0.9202, bal. acc 0.9005, acc 0.9609, MCC 0.9371, binary F1 0.9767, vehicle-level macro-F1 0.9541, ECE 0.0014, 2.5 ms per 1,000 messages (GPU predict).
6. **Leakage with rich features:** F1 0.908 → 0.954, F2 0.919 → 0.954 under a row split.
7. **SHAP:** bug found and fixed — `shap` sets XGBoost's `expected_value` only during the first `shap_values()` call (constant additivity error 2.69 before the fix). After fix: max additivity error 1.2e-5 on 3,000 stratified test messages. Perturbation: top-3 SHAP features replaced by genuine values → mean prob. drop 0.742, 79 % flips; 3 random features → 0.081, 9 %. Top global features: `dt`, `r_resid_absmean`, `r_dpos_mean`, `msg_index`, `r_spd_std`, `r_resid_std`, `r_dt_mean`, `r_vel_err_y_std`, `road_dist`, `r_same_frac`.
8. **Reports:** 200 stratified test messages (96 misclassified, 79 not high-confidence): template writer 200/200 pass validation (after shortening the template; 2 first-message reports had exceeded the 400-word limit). Median 0.10 s for features + model + SHAP per message on CPU. Validator false positives found and fixed through tests (feature names inside descriptions; class names inside definitions).
9. **LLM backend:** verified against a local mock OpenAI-compatible server only (success, corrective retry, validation fallback, network fallback). Real LLM run (2026-09-20, Groq `openai/gpt-oss-120b`, `outputs/results/report_faithfulness_summary.csv`): 15 reports, 15 written by the LLM, 14 passed validation first try, 1 after the corrective retry, 0 template fallbacks, mean 294 words (5 misclassified and 4 not-high-confidence messages in the sample). Small sample — a larger run and a by-eye read of the reports are still worthwhile.
10. **App:** headless AppTest passes (loads, generates a validated report); real server answered HTTP 200.
11. **SHAP vs. class definitions** (`outputs/figures/shap_per_class_heatmap.png`): the dominant feature of each class matches its definition — Constant speed → `r_spd_std`; Constant speed offset → `r_vel_err_x_mean` (stable bias); Constant position offset → `road_dist`; Random position / offset → `r_dpos_mean`; DoS → `r_same_frac`; Eventual stop → `acl`, `spd`; Delayed messages → `hedy_n` (the zero-filled warm-up rows); DoS/Sybil variants → `dt`, `r_dt_mean`; Data replay Sybil → `msg_index` (fresh pseudonyms). No identifier or absolute-time artefacts.
12. **Notebook** `VeReMi_XAI_Framework.ipynb` repaired (portable root, `behavioral_features` defined, real gap check, single-pass constant-column scan, heading removed from the noise-to-signal ratio) and extended with a Phases 4–8 walk-through. Executed top-to-bottom: 34 code cells, 0 errors; regenerated `overall_feature_statistics.csv`, `class_wise_feature_means.csv` and the EDA figures are byte-identical to the originals (the originals were kept in a local backup that is not part of the repository).

## 2026-09-20/21 — First real LLM evaluation (free Groq key added)

Setup: Groq free tier, OpenAI-compatible endpoint, temperature 0, `reasoning_effort: low`. `llama-3.3-70b-versatile` no longer exists on Groq (HTTP 404 → clean template fallback, as designed); models available to the key were listed through `/models` and the profiles were set to `openai/gpt-oss-120b` (`groq`) and `openai/gpt-oss-20b` (`groq_small`). Measured free limits for gpt-oss-120b: 8,000 tokens/min, 200,000 tokens/day, 1,000 requests/day; one report ≈ 2.4–3 k tokens.

1. **Prompt v1 + validator v1, gpt-oss-120b, 39 reports:** 15 accepted, 24 rejected → template fallback worked every time. Reading the rejected drafts showed that **20 of the 24 rejections were validator false positives** caused by LLM typography: thousands separators with a narrow no-break space ("57 596.435"), scientific notation rewritten as "1.43 × 10⁻¹²", non-breaking hyphens as minus signs, "82.1 %" with a space, back-ticked JSON keys (`senderPseudo`), lower-case descriptive phrases equal to a class name ("random position offsets"), and markdown table symbols counted as words. **4 were genuine violations and were correctly caught:** numbers computed by the LLM ("58 000 s + 0.269 s", "13 %", "46 %") and one over-long report. Re-validating the same 39 drafts offline with the fixed validator: 34 pass, 4 fail (1 not matchable).
2. **Fixes:** validator `normalise()` (typography), JSON keys / allowed class names in back-ticks, case-sensitive foreign-class detection, real-word count; evidence values < 1 rounded to 4 decimals (no scientific notation); **prompt v2** (copy numbers verbatim, no tables, 250-word limit, "pseudonym" not "vehicle"); backend honours the provider's retry hint and stops on daily quota; per-profile `max_tokens`/`extra` options in `config.yaml`. New regression tests for every case.
3. **Prompt v2 + validator v2 (stratified test sample incl. misclassified and low-confidence cases):**
   - `openai/gpt-oss-120b`: **15/15** reports written by the LLM and valid (14 first try, 1 after the corrective retry, 0 fallbacks). Stopped by the daily token quota (most of it was spent on the v1 run); the remaining 45 of the 60-report sample are pending.
   - `openai/gpt-oss-20b`: **60/60** valid (55 first try, 5 after retry, 0 fallbacks); mean 257 words; median 1.6 s per report.
   - Template baseline: 200/200 valid.
4. **Binding check** (`scripts/06_binding_check.py`, offline): 749 report lines that name exactly one top feature (627 + 122) — **0** lines where another feature's value/SHAP appears without the feature's own.
5. Spot reading: low-confidence cases are reported as such with their alternatives (e.g. "Constant position … confidence 0.3559 (low) … Random position offset 0.3387").

## 2026-09-21 — Uncertainty, robustness, error causes (`scripts/07_uncertainty_and_errors.py`, `08_seed_robustness.py`; test split only)

1. **Bootstrap CIs** (whole vehicles resampled within each class, because the messages of one vehicle are not independent): final model macro-F1 0.9202 [0.9146, 0.9258], accuracy 0.9609 [0.9576, 0.9644], balanced accuracy 0.9005 [0.8942, 0.9072], MCC 0.9371 [0.9318, 0.9425] (`test_metrics_ci.csv`). Weakest classes: Data replay Sybil F1 0.634 [0.602, 0.662], Delayed messages 0.811 [0.769, 0.855], DoS disruptive Sybil 0.819 [0.788, 0.846], Disruptive 0.833 [0.788, 0.874] (`final_per_class_metrics_ci.csv`).
2. **Paired bootstrap of macro-F1 differences** (`paired_comparisons.csv`; every difference positive in 100 % of resamples): history features F1 − F0 = +0.539 [0.528, 0.552]; road map F2 − F1 = +0.0107 [0.0085, 0.0131]; pseudonym-keyed history − true-sender-keyed history = +0.061 (F2) and +0.062 (F1).
3. **Vehicle-level macro-F1 is reported twice with two values:** 0.9541 (model card, `evaluate.sender_level`) and 0.9567 (bootstrap table). Both are a majority vote of a vehicle's message predictions; the code of the two differs in how vote ties are broken (`value_counts().index[0]` vs `argmax`), the only difference found in the implementations. Recomputed from the saved test predictions on 2026-09-21: 0.9541 with the first rule, 0.9567 with the second; the two rules disagree on 3 of the 3,710 test vehicles (those with a tied vote). The CI in `test_metrics_ci.csv` belongs to the second value.
4. **Split-seed robustness:** the final configuration retrained on three other class-stratified vehicle splits (seeds 1, 2, 3; 3,710 test vehicles each): test macro-F1 0.9203 / 0.9189 / 0.9194, balanced accuracy 0.9006 / 0.8994 / 0.8987; with the official split (seed 42) the mean is 0.9197, std 0.0007.
5. **Confidence levels printed in the reports** (`confidence_level_validation.csv`): *high* = 92.9 % of messages, accuracy 0.9782, 52.0 % of all errors; *moderate* = 5.5 %, accuracy 0.7893, 29.5 % of errors; *low* = 1.6 %, accuracy 0.5557, 18.5 % of errors. So the level is informative (accuracy falls with it) but about half of the errors are made at *high* confidence, so their reports state a high confidence level. Selective prediction (`selective_prediction.csv`): keeping confidence ≥ 0.9 covers 92.9 % of messages with accuracy 0.9782, macro-F1 0.9625 and removes 48.1 % of the errors; ≥ 0.99 covers 26.3 % with accuracy 0.996.
6. **Explanations of the main error types were tested, not assumed** (`error_cause_checks.json`):
   - Eventual stop: 1,109 of 5,995 messages are errors, all predicted Genuine; error rate 1.00 while the vehicle is still moving vs 0.014 once it has stopped.
   - Grid Sybil: 96.3 % of the 4,150 errors lie in one pseudonym per vehicle (error rate inside it 0.668; the other pseudonyms 0.215); that pseudonym lies on the road map like the others (median distance 1.96 m vs 1.99 m) and its speed-vs-position disagreement is small (median 0.37 vs 12.4 in the other pseudonyms), which is consistent with genuine-looking beacons of the attacker's own identity.
   - Delayed messages: the 56.3 % all-zero warm-up rows are never wrong; the other rows are wrong in 71.4 % of cases (99.9 % of errors predicted Genuine).
   - Constant position offset: error rate 0.002 when the offset position is off the road map (80 % of rows) vs 0.647 when it still lands on a road.
   - First message of a pseudonym: 3.7 % of test messages, error rate 0.268 vs 0.030 for later messages, 25.6 % of all errors; first messages make up 50 % / 41 % / 43 % of Data replay Sybil / DoS random Sybil / DoS disruptive Sybil (fresh pseudonyms) but only about 1 % of Genuine and Grid Sybil.
   - Data replay Sybil is predicted as DoS disruptive Sybil in 34.6 % of cases (recall 0.531); the reverse in 5.3 %.

## 2026-09-21 — Tables, showcase, report completeness, app screenshots, diagram, rating sheet

- `scripts/09_paper_tables.py`: tables T0–T11 (+ T10b completeness, T10c binding check) as CSV and Markdown in `outputs/results/paper_tables/`; only the literature rows of T11 are typed (marked as reported by the authors). T10 was regenerated after the gpt-oss-120b run was completed.
- **gpt-oss-120b evaluation finished** (same 60-message sample, prompt v2, validator v2): 59/60 written by the LLM and valid (55 first try, 4 after the corrective retry), 1 template fallback (the corrective retry still named the feature `spdy_n`, which is not in that message's evidence — the validator caught it). Mean 304 words, median 4.3 s per report (gpt-oss-20b: 257 words, 1.6 s). Binding check: 0 suspicious lines out of 441 (120b) and 627 (20b).
- `scripts/11_report_quality.py` (objective completeness rules, no LLM judge): top SHAP feature named 100 % / 100 % (120b / 20b); confidence level stated 100 % / 93.3 %; runner-up class named when confidence is not high 94.7 % (18 of 19) / 100 % (20 of 20); predicted class in the summary 100 % / 100 %; simulated-data caveat 100 % / 100 %.
- `scripts/10_showcase.py`: 14 curated test messages (8 correct, 6 errors) in `models/showcase_messages.json`, their local SHAP figures, and cached gpt-oss-20b reports so the demo runs without internet or quota.
- `scripts/12_app_screenshots.py`: five app screenshots in `outputs/figures/app/` (the dropdown list of the app is virtualised, so the script types the option text before selecting it).
- `scripts/13_architecture_diagram.py`: `outputs/figures/architecture.png`.
- `scripts/14_human_rating.py`: blind, paired rating sheet — 15 test messages × (gpt-oss-20b report, template report) = 30 shuffled items, 5 of the messages misclassified; rating columns clarity 1–5, usefulness 1–5, "anything invented" Y/N. **Not rated yet**: no claim about clarity or usefulness is made until it is.
- **Part 2 notebook executed** (`VeReMi_XAI_Part2.ipynb`): 33/33 code cells, 0 errors; the live-computed metrics of the final model agree with the model card; the Groq reports come from the cache (no API call).
- **Repository prepared for publication:** documentation moved to `docs/` (English; `PROJECT_GUIDELINES.md` trimmed to `docs/background.md`, sections renumbered); `LICENSE` (MIT), `NOTICE.md` (the dataset is CC BY 4.0, checked against its Zenodo record) and `data/README.md` (source, size, SHA-256) added; the dataset-dependent tests now skip cleanly when `artifacts/` is missing (found by running the tests on a copy that contains only the files to be published); the old walk-through cells of the Part 1 notebook (Part 2 replaces them) and local absolute paths were removed; the dataset audit script was renamed `dataset_audit.py`.

## 2026-09-21 — Two checks made while documenting the project

- **Shared placeholder pseudonym.** `senderPseudo == 1` is a placeholder shared by many vehicles, all of class 16 (Grid Sybil). Because the history features are grouped by pseudonym, the history of these rows mixes several vehicles. Measured on the test set (saved predictions): 5,083 of 478,186 test rows (1.06 %, belonging to 58 of the 59 class-16 test vehicles) carry this pseudonym, and they are almost always classified correctly (recall 0.9996, versus 0.7852 for the other class-16 rows). Removing them from the test set lowers the overall macro-F1 from 0.9202 to 0.9188 and the class-16 F1 from 0.9038 to 0.8756. So the headline number hardly depends on this artefact (−0.0014), but part of the Grid Sybil score does; it is a limitation of the dataset (the placeholder occurs only in class 16) that a real receiver would not have. The models were not retrained without it.
- **Column count.** The working table has 21 columns (4 identifiers, the label and 16 features): 30 columns minus `type` and the 8 z-columns. Table T0 and one notebook cell had said 22; corrected.
