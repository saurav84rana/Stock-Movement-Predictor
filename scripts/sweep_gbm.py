from src.models.evaluation import parameter_sweep, trim_tail, summarise_by_column
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RESULTS_RAW_DIR = ROOT_DIR / "results" / "raw"
RESULTS_PROCESSED_DIR = ROOT_DIR / "results" / "processed"

RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gradient_boosting"

df = pd.read_csv(PROCESSED_DIR/ "processed.csv", parse_dates=["Date"]
                 ).reset_index(drop=True)

cutoff = pd.to_datetime("2025-01-01")
dev_df = trim_tail(df[df["Date"] < cutoff], gap=25)

param_grid = [
    {},  # default parameters
    {"max_iter": 300}, {"max_iter": 1000},
    {"learning_rate": 0.05}, {"learning_rate": 0.2},
    {"max_leaf_nodes": 7}, {"max_leaf_nodes": 63},
    {"min_samples_leaf": 100}, {"min_samples_leaf": 200}
]

results = parameter_sweep(dev_df, MODEL_NAME, param_grid)
results.to_csv(RESULTS_RAW_DIR / "ps_gbm.csv", index=False)

summarise_by_column(results).to_csv(
    RESULTS_PROCESSED_DIR / "ps_gbm_summary.csv", index=False)
