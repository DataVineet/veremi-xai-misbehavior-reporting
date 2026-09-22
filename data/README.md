# data/

The dataset is **not** part of the repository (1.2 GB). To run everything that needs it (scripts `01`–`04`, `07`–`10`, the audits, the notebooks, the dataset-dependent tests), put the file here as

```
data/mixalldata_clean.csv
```

## Where to get it
- Kaggle mirror used by this project: <https://www.kaggle.com/datasets/ivarprudnikov/veremi-extension-data-1-21-gb> (file `mixalldata_clean.csv`).
- Original dataset (CC BY 4.0): VeReMi Extension, <https://zenodo.org/records/20090854>, DOI 10.5281/zenodo.20090854. See [NOTICE.md](../NOTICE.md) for the citation.

The cleaning that produced the Kaggle file is not documented anywhere. The file was checked against the paper: its vehicle counts (7,399 misbehaving + 17,264 genuine) match the "MixAll 0024" row of the paper's Table I exactly (`docs/background.md` §2.1).

## How to check that you have the same file
| Property | Value |
|---|---|
| Size | 1,213,467,430 bytes |
| SHA-256 | `884e4567c39c1ee2560e09c48b7c6319baffab5a8ba5ff6149b071f8bfad9f1d` |
| Rows × columns | 3,194,808 × 30 |
| Vehicles / pseudonyms / classes | 24,663 / 118,909 / 20 |

```powershell
Get-FileHash data\mixalldata_clean.csv -Algorithm SHA256
```

The same numbers are written to `outputs/results/paper_tables/T0_dataset_overview.md` by `scripts/09_paper_tables.py`. If your file differs, the results may differ too.

## Then
`python scripts/01_prepare_data.py` and `python scripts/01b_road_feature.py` create the intermediate files in `artifacts/`. Everything else is described in [docs/running_guide.md](../docs/running_guide.md).
