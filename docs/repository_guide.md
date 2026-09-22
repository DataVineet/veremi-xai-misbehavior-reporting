# Repository guide

A file-by-file explanation of the repository. How to run things is in [running_guide.md](running_guide.md) and the [README](../README.md); the reasons behind the design are in [architecture_decisions.md](architecture_decisions.md); dated measurements are in [research_log.md](research_log.md); the dataset facts, class-mapping evidence and literature are in [background.md](background.md).

---

## 0. The project in one paragraph

V2X messages of vehicles come in → an ML model says whether a message is **genuine or one of 19 misbehaviours** → SHAP says **why** the model decided so → the facts are packed into a small JSON, the *evidence package* → an LLM (or a template) turns that JSON into a **human-readable report** → a checker verifies that the report contains nothing that is not in the evidence → everything is shown in a Streamlit app.

```
data/ CSV ─► scripts/01, 01b ─► artifacts/ (features) ─► scripts/02 (training) ─► outputs/results/experiments
                                                               │
                                    scripts/03 (final model + SHAP) ─► models/ ─► app/streamlit_app.py
                                                               │
                        scripts/04 (figures) ─► outputs/figures        scripts/05 (reports) ─► outputs/reports
```

All logic lives in the package `src/veremi_xai/`. The scripts, the notebooks and the app **import that package**, so the same code runs everywhere.

---

## 1. Repository root

| File / folder | What it is |
|---|---|
| `README.md` | Front page: headline results, quick start, how to reproduce everything |
| `LICENSE`, `NOTICE.md` | MIT licence of the code; attribution and licence of the data the repository builds on |
| `config.yaml` | **Single source of truth**: paths, seed (42), names + category + definition of the 20 classes, LLM provider profiles |
| `requirements.txt` | Pinned package versions |
| `.env.example` | Template for API keys. Copy it to `.env` and paste a key to enable LLM-written reports (`.env` is git-ignored) |
| `VeReMi_XAI_Framework.ipynb` | Notebook Part 1: data understanding, data quality and exploratory analysis (section 7) |
| `VeReMi_XAI_Part2.ipynb` | Notebook Part 2: vehicle-level structure → features → models → uncertainty → SHAP → reports → validator → app (section 7) |
| `docs/` | Documentation (this folder) |
| `src/`, `scripts/`, `app/`, `tests/` | Code (sections 3, 4, 5) |
| `models/`, `outputs/` | Deployable model files and generated results (section 5) |
| `data/` | Place of the dataset CSV (not in the repository; see `data/README.md`) |
| `artifacts/` | Large intermediate files; created by the scripts, git-ignored |

### `docs/`

| File | Content | Open it when |
|---|---|---|
| `background.md` | Objective, verified dataset facts, class-mapping evidence, methodological foundations, literature review | You need the evidence behind a claim about the data or the class mapping |
| `architecture_decisions.md` | The design decisions AD-01 … AD-18, each with its measured evidence | You ask "why was it done this way?" |
| `research_log.md` | Dated log of what was measured, with the numbers | You need the source of a number |
| `repository_guide.md` | This file | You want to know what a file does |
| `running_guide.md` | Step-by-step run and check guide | You want to run or verify something |

---

## 2. `data/`

`mixalldata_clean.csv` (not included): 1.2 GB, 3,194,808 messages from 24,663 vehicles, 20 classes; the CSV built from the VeReMi Extension "MixAll 0–24 h" scenario. It is never modified; all scripts only read it. Where to obtain it and how to check the file: `data/README.md`.

---

## 3. `src/veremi_xai/` — the Python package

| File | Purpose | In plain words |
|---|---|---|
| `config.py` | Loads `config.yaml`, resolves paths, reads `.env` | Every other file takes its settings from here |
| `data.py` | CSV → Parquet conversion; builds the **vehicle-level split** | Divides the vehicles 70/15/15 (train/val/test), class-stratified. All messages of one vehicle go into one partition, so there is no leakage |
| `features.py` | Feature sets **F0** (18 per-message features), **F1** (+22 history features), **F2** (+`road_dist`); a plain-language `DESCRIPTIONS` dictionary; the `RoadMap` class | History features are causal: they look only at earlier messages. `road_dist` is the distance of the message position from the roads driven by genuine vehicles |
| `models.py` | Model factory: Logistic Regression, Decision Tree, Random Forest, XGBoost (on GPU if available) | All models are created from one place with the same settings |
| `evaluate.py` | Macro-F1, balanced accuracy, MCC, per-class table, confusion matrix, vehicle-level metric, calibration (ECE) | Accuracy is never reported alone |
| `explain.py` | `ShapExplainer`: exact TreeSHAP on the trained model | Contains the fix for reading SHAP's base value only after the first call |
| `evidence.py` | Builds the **evidence package** (JSON): prediction, confidence level, top SHAP features with their actual values and meanings, class definition, limitations | This JSON is all the report writer gets. It contains neither the ground truth nor the true sender id |
| `pipeline.py` | `Pipeline`: raw messages → features → prediction → SHAP → evidence | The app, the evaluation scripts and the notebooks all call this |
| `viz.py` | Chart colours and style | Makes all figures look alike |

### `src/veremi_xai/reporting/` — the report-writing layer

| File | Purpose |
|---|---|
| `__init__.py` | `generate_report(evidence, backend)`: the main entry point. Chooses the backend, checks the cache, calls the LLM, validates the text, retries once, and **falls back to the template** if the text fails validation twice or the provider is unavailable |
| `prompts.py` | The strict rules for the LLM: do not change the prediction, use only the evidence, five fixed headings, at most 250 words |
| `template_backend.py` | Deterministic report writer without an LLM; always works, also offline |
| `openai_compat.py` | One HTTP client for every provider that speaks the OpenAI format (Groq, Gemini, OpenRouter, Ollama) |
| `validator.py` | The **faithfulness checker**: headings correct? predicted class named? no foreign class? no language disputing the prediction? every number present in the evidence? every feature name in the evidence? length within the limit? |
| `base.py` | The `Report` data structure (text, writer, validation result, fallback reason) |

---

## 4. `scripts/` — steps that run in numeric order

| Script | What it does | Needs the dataset? | Output |
|---|---|---|---|
| `dataset_audit.py` | Full audit of the CSV: rows, classes, senders, gaps, duplicate vectors, class signatures | yes | print only (numbers of `background.md` §3–4) |
| `class_mapping_verification.py` | Data tests of the identity of every class (the proof of the class mapping) | yes | print only (`background.md` §4.5) |
| `01_prepare_data.py` | CSV → Parquet, vehicle split, features (history keyed on pseudonym and on sender) | yes | `artifacts/` |
| `01b_road_feature.py` | Road map from genuine **training** traffic and the `road_dist` feature | yes | `artifacts/road_dist.parquet`, `road_cells.npy` |
| `02_train_models.py` | 20 experiments: model comparison, feature ablation, leakage demonstration, class weighting, small hyper-parameter check | yes | `outputs/results/experiments/*.json` |
| `03_finalize_and_explain.py` | Freezes the final model, writes the model card and the demo data, global SHAP and sanity checks | yes | `models/`, `outputs/results/shap_*` |
| `04_make_figures.py` | Result figures and summary tables | yes | `outputs/figures/`, `outputs/results/*.csv` |
| `05_evaluate_reports.py` | Reports on stratified test messages and their faithfulness evaluation (`template` or an LLM backend) | no | `outputs/reports/`, `report_faithfulness_summary.csv` |
| `06_binding_check.py` | Offline check that a number in a report line is attached to the right feature | no | `outputs/results/binding_check.json` |
| `07_uncertainty_and_errors.py` | Bootstrap confidence intervals (vehicles resampled), paired comparisons, calibration, confidence-level validation, tests of the causes of the errors | yes | `outputs/results/*_ci.csv`, `error_cause_checks.json`, figures |
| `08_seed_robustness.py` | Retrains the final configuration on three more vehicle splits | yes | `outputs/results/robustness/` |
| `09_paper_tables.py` | Tables T0–T11 (CSV + Markdown) and `class_signatures.png` | yes | `outputs/results/paper_tables/` |
| `10_showcase.py` | The "Quick examples" of the app (14 curated test messages), their SHAP figures, cached reports | yes | `models/showcase_messages.json`, `outputs/figures/local_explanations_*.png` |
| `11_report_quality.py` | Completeness of the reports by fixed rules (no LLM judge) | no | `outputs/results/report_quality_summary.csv` |
| `12_app_screenshots.py` | Screenshots of the running app (optional: needs `playwright` and Edge/Chrome) | no | `outputs/figures/app/` |
| `13_architecture_diagram.py` | Block diagram of the whole pipeline | no | `outputs/figures/architecture.png` |
| `14_human_rating.py` | Builds the blind, paired rating sheet (LLM vs template) and summarises a filled sheet | no | `outputs/human_rating/` |
| `check_llm.py` | Quick check of an LLM provider with one report | no | print only |

---

## 5. `models/`, `outputs/`, `tests/`, `app/`

**`models/`** (small; the app runs from it)
- `final_model.ubj`: the trained XGBoost detector · `model_card.json`: model description and metrics
- `road_cells.npy`: the road map · `demo_messages.parquet`: 88,058 messages of 533 **test** vehicles (for the app; a subset of the CC BY 4.0 dataset, see `NOTICE.md`) · `showcase_messages.json`: the curated examples

**`outputs/`**
- `figures/`: the exploratory figures of notebook Part 1 and the result figures (`experiment_comparison`, `final_confusion_matrix`, `final_per_class_f1` and its `_ci` version, `shap_global_importance`, `shap_per_class_heatmap`, `calibration_reliability`, `selective_prediction`, `confidence_levels`, `class_signatures`, `local_explanations_*`, `architecture.png`, and `app/` with screenshots)
- `results/`: summary CSVs, SHAP tables and sanity checks, confidence intervals, error-cause checks, `report_faithfulness_summary.csv`, `experiments/` (the full JSON of every experiment), `robustness/`, and `paper_tables/` (tables T0–T11 as CSV and Markdown)
- `reports/`: generated reports (`reports_<writer>.jsonl`) and their evaluation
- `human_rating/`: the blind rating sheet (30 items) and its instructions

**`tests/`**
- `test_reporting.py`: validator, mock LLM server, fallback
- `test_features.py`: features do not look ahead in time (needs `artifacts/`; skipped otherwise)
- `test_data_and_evidence.py`: split (needs `artifacts/`; skipped otherwise), class mapping, evidence package
- `test_pipeline_and_app.py`: pipeline equals the batch results, SHAP is additive, the app loads headlessly

**`app/streamlit_app.py`**: four tabs — *Detect & explain* (choose a message → prediction → SHAP → evidence → report), *Model performance*, *Global explainability*, *About & limitations*. It contains no modelling logic; everything comes from the package.

---

## 6. Glossary

| Term | Meaning |
|---|---|
| `sender` vs `senderPseudo` | `sender` is the simulator's real vehicle id (nobody sees it in the real world); `senderPseudo` is the identifier a vehicle broadcasts, which is what a receiver sees. A Sybil attacker uses many pseudonyms |
| Leakage | Messages of the same vehicle in both train and test: the model memorises vehicles and the score is falsely high. The remedy used here is the vehicle-disjoint split |
| F0 / F1 / F2 | Feature sets: the message only (macro-F1 0.37) → + history (0.91) → + road map (0.92) |
| Macro-F1 | The mean of the per-class F1 scores; small classes count equally. More honest than accuracy on imbalanced data |
| SHAP value | How much a feature pushed the model's score up or down for this message. Base value + all SHAP values = the model output |
| Evidence package | The small JSON given to the report writer; nothing outside it may appear in a report |
| Faithfulness validation | The automatic check that a report stays inside the evidence |
| Fallback | If the LLM is unavailable or writes an invalid text, the template report is used and the reason is stored |

---

## 7. The notebooks

### `VeReMi_XAI_Framework.ipynb` — Part 1 (63 cells, 28 code cells)

Data understanding, data quality and exploratory analysis. It runs top to bottom without errors (about 20 minutes, because it reads the 1.2 GB CSV in chunks about 14 times; memory use stays low). The outputs are saved in the notebook, so it can simply be read.

**Phase 1 — Data understanding (cells 0–16)**
- Cell 1: imports and paths (portable, relative to the working directory), chunk size 200,000. Cell 3: `process_chunks()`, the helper that reads the CSV in pieces.
- Cells 5–6: the first five rows, the 30 columns, the meaning of one record.
- Cells 8–9: a full scan → **3,194,808 records, 20 classes, 24,663 senders, 118,909 pseudonyms, all messageIDs unique**.
- Cells 11–16: the class ids 0–19 are verified; class distribution table and bar chart (`class_distribution.png`); class 0 = 59.49 % (imbalance).

**Phase 2 — Data quality and integrity (cells 17–33)**
- Cell 18: missing values → none. Cell 20: exact duplicate rows (by row hash) → none.
- Cells 23–24: classes per sender → **every sender has exactly one class**, so a random split would leak (the most important structural fact of the project).
- Cells 26–27: pseudonym audit: most senders have one pseudonym, some up to 100; `senderPseudo = 1` is shared by 377 senders.
- Cell 29: `sendTime` spans 240 s – 86,400 s (a full day); the file is not globally time-sorted.
- Cells 31–32: message gaps within a sender: median 1 s, 77 % of the gaps are 1 s, minimum 0.167 s, **no zero or negative gap**.
- Cell 33: validation checkpoint summary.

**Phase 3 — Exploratory data analysis (cells 34–61)**
- Cells 35–37: feature inventory; the **constant columns** (`type` and the eight z-columns) and the unique column (`messageID`).
- Cell 39: feature groups and the definition of `behavioral_features` (16 features).
- Cells 41–43: overall mean/std/min/max → `overall_feature_statistics.csv`.
- Cells 45–47: class-wise means → `class_wise_feature_means.csv`; classes 5, 7, 14, 18 have very high speed.
- Cells 49–53: bar charts of average speed / acceleration / heading by class.
- Cell 55: class-wise average absolute noise.
- Cells 56–57: noise-to-signal ratio for position, speed and acceleration (heading is excluded: `hedx_n` is on a ~0–120 scale and `hedx` on −1…1, so the ratio would be meaningless).
- Cells 59–61: a sample of 1,000 rows per chunk → class-wise boxplots of speed, acceleration and heading.
- Cell 62: pointer to Part 2.

### `VeReMi_XAI_Part2.ipynb` — Part 2 (63 cells, 33 code cells)

Reads the saved results and runs two small live experiments (a few minutes in total). All cells were executed without errors; the live metrics of the final model agree with the model card. Groq reports come from the cache, so no API call is needed.

| Step (title in the notebook) | Content |
|---|---|
| 1 · The data per vehicle | One class per vehicle, message rate, pseudonyms, class signatures (table and figure), and how the class identities were established |
| 2 · Splitting the data so that the test score is honest | The vehicle-disjoint split and a live experiment on the size of the leakage effect |
| 3 · Feature engineering | Causal history features (a worked example and a causality check), the road map (F2), which features separate which classes |
| 4 · Models | The 20 experiments and what they show |
| 5 · The final model | Live evaluation on the test vehicles, uncertainty (bootstrap over vehicles, other splits, calibration, confidence levels), the causes of the errors, the literature context |
| 6 · Explaining the predictions with SHAP | Global importance, per-class heatmap, one local explanation, faithfulness checks |
| 7 · From explanation to report | The evidence package, the template report, an LLM report, a case where the model is unsure and wrong |
| 8 · Can we trust the reports? | The validator on deliberately broken reports, faithfulness and completeness of the writers |
| 9 · The Streamlit app | Screenshots |
| 10 · Summary and limitations | Headline numbers and the limitations |

To run a notebook again: open it in VS Code, select the `.venv` kernel and choose "Run All".

---

## 8. Which question → which file

| Question | File |
|---|---|
| How do I run it? | `README.md`, `docs/running_guide.md` |
| What are the results and limitations? | `README.md` (headline results), `docs/research_log.md` |
| Why was this decision taken? | `docs/architecture_decisions.md` |
| Where does this number come from? | `docs/research_log.md` → `outputs/results/` |
| What is the evidence for the class mapping? | `docs/background.md` §4 and `scripts/class_mapping_verification.py` |
| What does the literature say? | `docs/background.md` §6 |
| What does feature X mean? | `src/veremi_xai/features.py` → `DESCRIPTIONS` (also shown in the app) |
| How do I change the LLM provider? | `config.yaml` → `llm.profiles`, and `.env` |
| How do I change the report rules? | `src/veremi_xai/reporting/prompts.py` (and raise `prompt_version` in `config.yaml`) |
