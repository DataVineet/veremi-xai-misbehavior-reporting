# Running guide

Step-by-step instructions to install, run and check the project. Each step says **what to type → what you should see → what to do if it fails**. The commands are for Windows PowerShell; the project was developed and tested on Windows with Python 3.12. The trained model and all results are in the repository, so **nothing has to be trained again** to use it.

What works without the dataset (a fresh clone): the tests of the reporting layer and of the pipeline/app, the app, the report evaluation and quality scripts (`05`, `06`, `11`), the diagram and the rating sheet (`13`, `14`). What needs `data/mixalldata_clean.csv`: everything that creates or reads `artifacts/` (scripts `01`–`04`, `07`–`10`, the audits, `test_features.py`, the split test, the notebooks).

---

## Step 1 — Install

```powershell
git clone <repository-url>
cd <repository-folder>
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

You should see `(.venv)` at the start of the prompt.

- "running scripts is disabled on this system": run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` and activate again.
- The environment has to be activated in **every new terminal**. Without `(.venv)` you will get `ModuleNotFoundError`.
- `playwright` in `requirements.txt` is only needed for `scripts/12_app_screenshots.py`.

---

## Step 2 — Run the tests (a few minutes)

Run them one by one and read the **last line**. Yellow warnings (XGBoost "mismatched devices", Streamlit `use_container_width`, `SyntaxWarning`) are harmless.

| Command | Expected last line | What it proves |
|---|---|---|
| `python tests/test_reporting.py` | `all reporting tests passed` | The validator accepts good reports and catches wrong numbers, classes and features; an unavailable or invalid LLM falls back to the template |
| `python tests/test_pipeline_and_app.py` | `pipeline ok` and `app ok` | Prediction → SHAP → evidence → report is consistent, SHAP is additive, the app loads without errors |
| `python tests/test_data_and_evidence.py` | `all data/evidence tests passed` (with `SKIPPED (split test)` if there is no `artifacts/`) | Class mapping and evidence package are correct; the split is vehicle-disjoint |
| `python tests/test_features.py` | `all feature tests passed`, or `SKIPPED: ...` without `artifacts/` | History features never look at future messages |

`SKIPPED` means that the dataset-derived files are missing: put the CSV into `data/` and run `python scripts/01_prepare_data.py` (about 2 minutes).

---

## Step 3 — Run the app

```powershell
streamlit run app/streamlit_app.py
```

(If `streamlit` is not recognised: `python -m streamlit run app/streamlit_app.py`.) You should see `Local URL: http://localhost:8501` and the browser opens by itself. The first load takes 10–20 seconds. Stop the app with `Ctrl + C` in the terminal.

Things to try in the tab **Detect & explain**:

1. **A normal case.** In "Quick example" choose "13 DoS: detected correctly". Below, the predicted class is *DoS* with a confidence of about 99 %. The SHAP chart is led by `r_same_frac` (0.8: most recent messages repeat their predecessor) and `dt` (0.25 s between messages), because DoS sends messages four times faster than normal.
2. **A report.** Scroll to "5 · Human-readable misbehavior report" and press **Generate report with `template`**. You get the five sections *Summary / Observed evidence / Model interpretation / Uncertainty and limitations / Suggested analyst action*, and below them a green box "Faithfulness validation passed". "Download report" and "Download evidence JSON" work as well.
3. **The evidence package.** Open "Show evidence JSON" in section 4. This JSON is all that the report writer receives. It contains neither the ground-truth class nor the true `sender` id.
4. **A wrong prediction.** Choose "17 Data replay Sybil: missed, predicted DoS disruptive Sybil". The prediction is shown with a low confidence and the alternative class next to it; the generated report says so in "Uncertainty and limitations". This is a documented limitation of the approach.
5. **What-if.** Choose the quick example "0 Genuine: detected correctly", open "What-if: edit the transmitted values of this message before scoring" and set `posx` to 5000. The prediction moves away from *Genuine* (for this example to *Disruptive* with a low confidence of about 45 %), and `r_resid_absmean`, the disagreement between the transmitted speed and the speed implied by the position jump, rises from about 0.07 to about 487 and leads the SHAP chart. Other messages give other numbers.
6. **Other tabs.** *Model performance*: macro-F1 0.920, per-class F1, confusion matrix, all 20 experiments, the leakage figure. *Global explainability*: top SHAP features, per-class heatmap, additivity error (about 1e-05), the probability drop when the top features are replaced (0.742 versus 0.081 for random features), report faithfulness. *About & limitations*: the pipeline, the 20 class definitions, the limitations.

The "Report writer" in the sidebar lists `template` always and an LLM profile only when its key is present in `.env` (Step 6).

---

## Step 4 — Look at the saved results (nothing to run)

- `outputs/figures/architecture.png` (pipeline), `experiment_comparison.png` (feature sets, models, leakage effect), `final_confusion_matrix.png`, `final_per_class_f1_ci.png`, `shap_global_importance.png`, `shap_per_class_heatmap.png`, `outputs/figures/app/` (app screenshots)
- `outputs/results/paper_tables/` — tables T0–T11 (CSV and Markdown), `outputs/results/test_metrics_ci.csv` — confidence intervals, `outputs/results/experiments_summary.csv` — the 20 experiments
- `outputs/reports/reports_template.jsonl` — 200 generated reports

---

## Step 5 — Verify the dataset facts (needs the dataset; about 1 minute each)

```powershell
python scripts/dataset_audit.py
python scripts/class_mapping_verification.py
```

The first prints the row count (3,194,808), one class per sender, message gaps and the per-class signature table. The second runs the tests of the class mapping (for example class 6: velocity error mean ≈ 7 with sd 0.73; class 17: pseudonym change on 99.2 % of the target changes). Together they are the evidence for the class mapping in `docs/background.md` §4.

---

## Step 6 — LLM-written reports (optional)

1. Copy `.env.example` to `.env` and paste **one** key, for example a free Groq key from console.groq.com. `.env` is git-ignored; never commit it.
2. Check the provider with one report: `python scripts/check_llm.py groq_small` → `WRITTEN BY: groq_small (model: openai/gpt-oss-20b)` and `validation passed: True`.
3. In the app choose `groq_small` (small, fast) or `groq` (large) as the report writer and press **Generate**. A finished report is cached, so repeating it is free.
4. Evaluate on the stratified sample (60 reports, roughly 35 minutes because the script paces the requests for the free tier): `python scripts/05_evaluate_reports.py groq_small 3` or `python scripts/05_evaluate_reports.py groq 3`. The free limit of `gpt-oss-120b` was measured as 8,000 tokens per minute and 200,000 tokens per day (about 70 reports per day); when it is exhausted the script stops and the same command can be repeated the next day — finished reports come from the cache.
5. Offline checks that need no API call: `python scripts/06_binding_check.py groq_small` (is every number attached to the right feature?) and `python scripts/11_report_quality.py` (completeness rules).
6. `HTTP 404 model not found`: the provider has renamed or removed the model. Look up the current name in the provider's console and edit `llm.profiles.<profile>.model` in `config.yaml`.

Without a key everything else works; the template writer is used.

---

## Step 7 — Human rating of the reports (optional)

`python scripts/14_human_rating.py make` writes `outputs/human_rating/rating_sheet.xlsx` (30 shuffled reports, the writer is hidden) and `rating_key.csv` (which report came from which writer; do not show it to the rater). Give the sheet to a rater — the sheet `instructions` explains what to do — and summarise the filled sheet with `python scripts/14_human_rating.py summarize <filled_sheet.xlsx>`. The sheet is reproduced exactly from the stored reports.

---

## Step 8 — Notebooks (optional; needs the dataset)

Open `VeReMi_XAI_Framework.ipynb` (Part 1, about 20 minutes because the 1.2 GB CSV is read in chunks) or `VeReMi_XAI_Part2.ipynb` (Part 2, a few minutes) in VS Code, choose the kernel `.venv` and select "Run All". No cell should turn red. The outputs are already saved in the notebooks, so reading them does not require a run.

---

## Step 9 — Reproduce everything from the raw CSV (needs the dataset)

Place `mixalldata_clean.csv` in `data/` (see `data/README.md`) and run in this order:

| Step | Command | Time |
|---|---|---|
| 1 | `python scripts/01_prepare_data.py` | ≈ 2 min |
| 1b | `python scripts/01b_road_feature.py` | ≈ 1 min |
| 2 | `python scripts/02_train_models.py` | **≈ 1.5 hours** (GPU optional) |
| 3 | `python scripts/03_finalize_and_explain.py xgb_F2_pseudo_d6` | a few min |
| 4 | `python scripts/04_make_figures.py` | a few min |
| 5 | `python scripts/05_evaluate_reports.py template 10` | ≈ 1 min |
| 7 | `python scripts/07_uncertainty_and_errors.py` | ≈ 4 min |
| 8 | `python scripts/08_seed_robustness.py 1 2 3` | ≈ 5 min per seed |
| 9 | `python scripts/09_paper_tables.py` | < 1 min |

The results of these steps are already in the repository. Do not re-run step 2 unless you want to reproduce the experiments.

---

## Common errors

| Error | Meaning | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named ...` | The virtual environment is not active | Step 1 |
| `running scripts is disabled on this system` | PowerShell policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| `streamlit : The term 'streamlit' is not recognized` | PATH | `python -m streamlit run app/streamlit_app.py` |
| `Port 8501 is already in use` | Another app instance is running | Stop it with `Ctrl + C`, or `streamlit run app/streamlit_app.py --server.port 8502` |
| `FileNotFoundError ... artifacts/messages.parquet` | The dataset-derived files have not been created | Put the CSV in `data/` and run `python scripts/01_prepare_data.py` |
| `FileNotFoundError ... models/...` | Files of `models/` are missing | Check that the clone is complete |
| The sidebar says "No API key found for: ..." | No key in `.env` | Normal: the template writer is used; see Step 6 |
| Many yellow warnings | XGBoost / Streamlit information | Ignore them; read the last line |
