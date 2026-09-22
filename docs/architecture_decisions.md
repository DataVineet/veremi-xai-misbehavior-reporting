# Architecture decisions

Each decision lists the evidence that drove it. Numbers are from `outputs/results/` (test set = vehicles never seen in training) unless stated. Append new decisions; never silently rewrite old ones — mark them superseded.

---

## AD-01 Class mapping = F2MD simulator enum (2026-09-20)
**Decision.** 0 Genuine … 19 DoS disruptive Sybil as in `config.yaml`. Fault/attack grouping per the VeReMi Extension paper (faults = 1–8, 12).
**Evidence.** Positive data tests for all 19 classes, F2MD `AttackTypes.h`, VeMisNet Table 3, MistralBSM Table I. Khan et al. (2025) Table 2 is contradicted for 14 classes. → `docs/background.md` §4, `scripts/class_mapping_verification.py`.

## AD-02 Sender-disjoint, class-stratified 70/15/15 split (2026-09-20)
**Decision.** Split by `sender`; file `artifacts/sender_split.csv` is reused by every experiment. A random row split is reported only as a leakage demonstration.
**Evidence.** Every sender has exactly one class; consecutive messages are near-identical. Measured inflation of macro-F1 by the row split: F0 0.369 → 0.680, F1 0.908 → 0.954, F2 0.919 → 0.954. The leaky F0 score (0.68) reproduces the ≈ 0.70 reported by Slama et al. on the same CSV, which supports the diagnosis.

## AD-03 Identifiers and absolute time are never model inputs (2026-09-20)
**Decision.** `messageID`, `sender`, `senderPseudo`, `sendTime` are excluded; only time *differences* are used.
**Evidence.** In Khan et al. the top SHAP feature is `messageID`; such "explanations" are artefacts.

## AD-04 Causal history features (F1) (2026-09-20)
**Decision.** 22 context features from the current and earlier messages of the same identity (gap, displacement, speed-vs-position and acceleration-vs-speed consistency, heading-vs-travel direction, repeats, rolling 10-message statistics). No look-ahead; computed **separately per data partition** so no test feature depends on training rows. Causality is unit-tested (`tests/test_features.py`).
**Evidence.** 22 % of rows are bit-identical to another row and 128 k genuine rows equal attack rows, so per-message features cannot work: F0 macro-F1 0.369 vs F1 0.908.

## AD-05 History is keyed on `senderPseudo`, not `sender` (2026-09-20)
**Decision.** Identity for history features = pseudonym (what a receiver can observe).
**Evidence.** It is more realistic **and** better: macro-F1 0.919 (pseudonym) vs 0.858 (true sender) with F2. Keying on the true sender hides the pseudonym switching that characterises Sybil attacks. `sender` is used only for splitting and for the vehicle-level metric.

## AD-06 Road-map plausibility feature (F2) (2026-09-20)
**Decision.** `road_dist` = distance to the nearest 5 m cell visited by **genuine training** vehicles (6,437 cells, `models/road_cells.npy`). A fitted preprocessing artifact, like a scaler.
**Evidence.** Error analysis of F1: 41 % of Constant-position-offset messages were predicted Genuine. Genuine test traffic is > 5 m off the map in 0.00 % of rows vs 80 % for class 2. Result: class-2 F1 0.738 → 0.929; macro-F1 0.908 → 0.919.

## AD-07 Detector = XGBoost (hist), no class weighting, no resampling (2026-09-20)
**Decision.** Final model `xgb_F2_pseudo_d6` (max_depth 6, lr 0.10, early stopping on validation mlogloss, 179 rounds kept).
**Evidence (F2, test macro-F1 / val macro-F1).** LogReg 0.800/0.789 · Decision tree 0.907/0.902 · Random forest 0.912/0.907 · XGBoost 0.919–0.921/0.914–0.916. Hyper-parameters barely matter (val 0.9142–0.9158 across four settings) → the model is at the data ceiling; among the near-ties the shallow model was chosen because TreeSHAP cost grows with depth and the artifact is smaller (selection used validation only). Balanced class weights raise balanced accuracy slightly (0.904 vs 0.901) but lower macro-F1 (0.908) and genuine precision → not used. SMOTE not used (synthetic kinematics are hard to justify; minority classes already have ≥ 30 k training rows).

## AD-08 Deployed model is sliced to the best iteration (2026-09-20)
**Decision.** `booster[: best_iteration + 1]` is saved as `models/final_model.ubj`.
**Evidence.** Identical probabilities, and it removes any ambiguity between `predict()` (best iteration) and tree-walking tools (all trees).

## AD-09 SHAP = `shap.TreeExplainer`, raw-margin space, predicted class (2026-09-20)
**Decision.** Exact TreeSHAP on the deployed model; local explanations use the SHAP vector of the predicted class; the evidence states the output space and base value explicitly.
**Evidence / pitfall found.** For XGBoost, `shap` sets the correct `expected_value` only during the first `shap_values()` call. Reading it earlier gave a constant additivity error of 2.69; fixed in `explain.py` and guarded by a test. Now max additivity error 1.2e-5 over 3,000 messages. Perturbation check: top-3 SHAP features replaced → mean probability drop 0.742 (79 % predictions flip) vs 0.081 (9 %) for 3 random features.

## AD-10 Evidence package is the only input of the report writer (2026-09-20)
**Decision.** JSON with prediction, confidence level, two alternatives, top-k SHAP contributions with actual values and plain-language feature descriptions, base value/model output, transmitted values, model card extract, limitations. No ground truth, no true sender id. Values are pre-rounded so that text can quote them exactly.

## AD-11 Reporting layer: template + one OpenAI-compatible backend + validator + cache (2026-09-20)
**Decision.** `generate_report(evidence, backend)`; backends are config profiles (Groq, Gemini-compat, OpenRouter, Ollama, …); keys only from `.env`; temperature 0; disk cache keyed by evidence hash + prompt version + provider + model. Every text is validated (structure, predicted class named and no foreign class, no override language, every number present in the evidence, only evidence features, length). One corrective retry, then **fallback to the template**; the rejected draft is kept for transparency.
**Evidence.** No API key was available at that point and the provider decision had to stay reversible; free-tier limits change often (`docs/background.md` §5.8.1). Verified without a key through a local mock OpenAI server (`tests/test_reporting.py`): success path, corrective retry, validation-failure fallback, network-failure fallback.
**Known limitation.** The validator checks that a number exists in the evidence, not that it is attached to the right feature.

## AD-12 Streamlit is a thin layer over `Pipeline` (2026-09-20)
**Decision.** The app calls `Pipeline.analyse()` and `generate_report()`; it recomputes features with the training feature code (also for what-if edits) and only shows test vehicles. A test asserts that the app path reproduces the batch predictions and runs the app headlessly.

## AD-13 Not done on purpose
Cross-identity linkage features (e.g. "another pseudonym at the same position") that might separate Data replay Sybil from DoS disruptive Sybil: judged over-engineering for this project; documented as future work. Sequence deep-learning models: no evidence they are needed at the measured ceiling, and they would weaken exact SHAP explanations.

## AD-14 Validator must normalise LLM typography; prompt v2 (2026-09-21)
**Decision.** `validator.normalise()` removes thousands separators (thin/no-break space, comma), converts "a × 10ⁿ" to e-notation and maps typographic hyphens/spaces to ASCII before numbers are compared; back-ticked JSON keys and allowed class names are accepted; foreign-class detection is case-sensitive (capitalised class names as in the evidence); only real words count towards the length limit. Evidence values below 1 are rounded to 4 decimals so no scientific notation reaches the writer. Prompt v2 demands verbatim numbers, no tables, ≤ 250 words, and "pseudonym" instead of "vehicle".
**Evidence.** First real LLM run: 24/39 drafts rejected, 20 of them false positives of exactly these kinds; 4 genuine violations (LLM-computed numbers, over-length) were correctly caught. After the change: gpt-oss-120b 15/15 and gpt-oss-20b 60/60 valid, 6 of 75 needing the corrective retry, 0 fallbacks; offline binding check 0/749 suspicious lines (state at the time of this decision; the completed 60-report evaluation is in `docs/research_log.md`, 2026-09-21).
**Trade-off.** Case-sensitive class matching would miss a foreign class written in lower case ("this could be a grid sybil attack"); accepted because descriptive lower-case phrases ("random position offsets") are far more common in real drafts, and the predicted class itself is still required.

## AD-15 LLM profiles carry provider-specific options; evaluation respects free-tier quotas (2026-09-21)
**Decision.** A profile may set `max_tokens` and `extra` (merged into the request body, e.g. `reasoning_effort: low` for reasoning models whose hidden reasoning tokens count against `max_tokens`). The backend honours "try again in …" hints for per-minute limits and gives up immediately on daily-quota errors; the evaluation script paces requests (30 s), never counts outages as report results, stops on a daily quota, and caches both accepted reports and twice-rejected outcomes (deterministic at temperature 0) so a re-run costs nothing.
**Evidence.** Measured Groq free limits for gpt-oss-120b: 8 k tokens/min, 200 k tokens/day; ≈ 2.4–3 k tokens per report. Provider model line-ups change (Llama 3.3 70B had been removed) — hence models are config, never code.

## AD-16 Uncertainty is measured by resampling vehicles, and split-seed robustness is checked (2026-09-21)
**Decision.** Confidence intervals come from a bootstrap that resamples whole test vehicles within each class (messages of a vehicle are strongly dependent, so a message-level bootstrap would give intervals that are too narrow); model comparisons use a paired bootstrap on the same resamples; the final configuration is additionally retrained on three other vehicle splits.
**Evidence.** Macro-F1 0.9202 [0.9146, 0.9258]; four splits give 0.9197 ± 0.0007; F2 − F1 = +0.0107 [0.0085, 0.0131]. The interval reflects test-vehicle sampling only, not training randomness beyond the three extra splits.
**Trade-off.** Only 59 vehicles per attack class in the test set: per-class F1 intervals are wide (e.g. Data replay Sybil 0.602–0.662).

## AD-17 The confidence level shown in reports is validated on the test set, not assumed (2026-09-21)
**Decision.** Reports state one of three confidence levels (*high*: predicted-class probability ≥ 0.90 and a gap of ≥ 0.50 to the runner-up; *moderate*: probability ≥ 0.60; otherwise *low*; see `evidence.py`); their meaning is checked against measured accuracy and the report writer must state the level.
**Evidence.** Accuracy 0.978 / 0.789 / 0.556 for high / moderate / low. 52 % of all errors nevertheless happen at *high* confidence; the reports of those errors state a high confidence level.
**Known limitation.** Confident errors are dominated by the structural cases (attacker messages that look genuine: Eventual stop before the stop, Grid Sybil's own beacons, Delayed messages); they cannot be flagged from the model's probability alone.

## AD-18 Report quality is measured with objective rules and a blind human rating, not with an LLM judge (2026-09-21)
**Decision.** Completeness is checked by fixed rules (`scripts/11_report_quality.py`); clarity and usefulness are left to a blind, paired human rating (LLM vs template report of the same message, shuffled, writer hidden). An LLM judge was not used because a language model scoring language-model text would add a second unvalidated component to the evaluation.
**Evidence.** Rules: top SHAP feature named 100 %, confidence level stated 93–100 %, runner-up named 95–100 %. The human rating has not been carried out yet (`outputs/human_rating/`).
