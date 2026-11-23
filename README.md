# Budget-Aware Adaptation of Encrypted Traffic Classifiers Under Temporal Drift

Code, results and figures for our CMPS 342 (Advanced Computer Networks) research project at the American University of Beirut, by Lin Noureddine and Aya Daou.

We study how the number of newly labeled flows affects the accuracy and cost of updating a LightGBM encrypted traffic classifier after drift. We compare no update (KEEP), lightweight updates (REFIT, LIGHT) and full retraining (FULL, FULL_W, FULL_W120) at label budgets of 2k, 10k, 50k and 250k flows, on two time chains of CESNET-TLS-Year22.

## Data

CESNET-TLS-Year22 (Zenodo record 10608607): https://zenodo.org/records/10608607
The raw data is not included here. `code/download_weeks.py` downloads only the weeks we use.

| Chain | Training week | Label week | Test weeks | Older week (recency control) |
| ----- | ------------- | ---------- | ---------- | ---------------------------- |
| 1     | 18            | 22         | 23-30      | 17                           |
| 2     | 36            | 40         | 41-48      | 35                           |

## Setup

```
pip install -r requirements.txt
```

Set the folders once (conda example):

```
conda env config vars set CESNET_RAW=<raw data folder> CESNET_OUT=<output folder> CESNET_PROJECT=<this repo>
```

## How to reproduce

Run from `code/`, in this order:

1. `python download_weeks.py`: download the weeks above
2. `python preprocess.py`: one parquet file per week
3. Notebooks `01` to `05` in `notebooks/`: class selection, learning curve (S = 1M), training and test samples
4. `python make_samples_l4.py`: label budget samples and the two control samples
5. `python run_grid.py <chain>`: main grid, then `--plan rounds`, `--plan capacity`, `--plan timing`, `--plan predict`
6. `python bootstrap_bytes.py <chain>` and `python bytes_breakdown.py <chain>`: traffic metrics with confidence intervals
7. `python paper_figures.py --dirs <results folder>`: all figures and tables

Timing and prediction runs should be done with other programs closed.

## Folders

- `code/`: scripts
- `notebooks/`: class selection, learning curve and sample creation
- `config/`: features, classes, label encoder and LightGBM settings (fixed before running)
- `samples/`: label yield tables and sample metadata (the `.npy` index files are not included; they are rebuilt from the fixed seeds)
- `results/`: one row per model and test week (`results_c1/c2.csv`), timing, prediction speed, bootstrap results
- `figures/`: figures and tables used in the paper
- `logs/`: run logs from both laptops

## Notes

- Library versions are pinned in `requirements.txt`; results were identical on both laptops.
- FULL_W120 was added after the first results, to compare LIGHT with a full retrain of the same size.

## License

MIT, see `LICENSE`.
