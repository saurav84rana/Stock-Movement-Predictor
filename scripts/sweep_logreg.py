from src.models.evaluation import parameter_sweep, trim_tail, summarise_by_column
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RESULTS_RAW_DIR = ROOT_DIR / "results" / "raw"
RESULTS_PROCESSED_DIR = ROOT_DIR / "results" / "processed"

RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "logistic_regression"

df = pd.read_csv(PROCESSED_DIR/ "processed.csv", parse_dates=["Date"]
                 ).reset_index(drop=True)

cutoff = pd.to_datetime("2025-01-01")
dev_df = trim_tail(df[df["Date"] < cutoff], gap=25)

param_grid = [
    {},  # default parameters
    {"C": 0.01}, {"C": 0.1}, {"C": 10}, {"C": 100}
]

results = parameter_sweep(dev_df, MODEL_NAME, param_grid)
results.to_csv(RESULTS_RAW_DIR / "ps_logreg.csv", index=False)

summarise_by_column(results).to_csv(
    RESULTS_PROCESSED_DIR / "ps_logreg_summary.csv", index=False)