# Notice: licences and attribution

## Code and documentation
The source code, scripts, notebooks and documentation of this repository are released under the MIT licence (see [LICENSE](LICENSE)).

## Data
This project is built on the **VeReMi Extension** dataset, which is licensed under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

> Joseph Kamel, Michael Wolf, Rens W. van der Heijden, Arnaud Kaiser, Pascal Urien and Frank Kargl,
> "VeReMi Extension: A Dataset for Comparable Evaluation of Misbehavior Detection in VANETs",
> IEEE International Conference on Communications (ICC), 2020.
> Dataset record: https://zenodo.org/records/20090854 (DOI 10.5281/zenodo.20090854).

The file used here, `mixalldata_clean.csv`, is a Kaggle mirror of the "MixAll" scenario
(<https://www.kaggle.com/datasets/ivarprudnikov/veremi-extension-data-1-21-gb>). It is **not** part of this repository.

What of the data this repository does contain, and what was changed:

- `models/demo_messages.parquet`: 88,058 messages of 533 test vehicles (a subset of the rows of the CSV; the 21 original columns are unchanged) plus three columns added by this project (`pred`, `conf`, `in_demo_pick`: the model's prediction, its confidence and a flag for the app). It lets the demo app run without the full dataset.
- `outputs/`, `models/final_model.ubj` and the notebooks contain statistics, figures, tables and a model derived from the data.

These files are shared under the same CC BY 4.0 terms as the data they derive from: please credit the authors above when you reuse them.

## Generated report texts
The texts in `outputs/reports/` and in the notebooks were written by language models through an API (`openai/gpt-oss-20b` and `openai/gpt-oss-120b`, served by Groq) or by the deterministic template in this repository, from the evidence packages produced by the pipeline. They are kept as evidence for the evaluation in `docs/research_log.md`.

## Third-party software
The project depends on open-source packages listed in `requirements.txt` (among them pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, matplotlib), each under its own licence.
