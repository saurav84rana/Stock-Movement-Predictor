import pandas as pd
from pathlib import Path
from src.models.train import fit_strategy, save_model

PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)

processed_df = pd.read_csv(PROCESSED_DIR / "processed.csv", parse_dates=["Date"])

NON_FEATURE_COLS = ["Date", "Open", "High", "Low", "Close", "Volume", "ticker", "target"]
FEATURE_COLS = [c for c in processed_df.columns if c not in NON_FEATURE_COLS]

# full dataset minus last day, deployment only
all_last_idx = processed_df.groupby("ticker")["Date"].idxmax()
train_df = processed_df.drop(index=all_last_idx)
X, y = train_df[FEATURE_COLS], train_df["target"]

name_filename = [
    ("logistic_regression", "production_logreg.joblib"),
    ("gradient_boosting", "production_gbm.joblib")
]
for name, filename in name_filename:
    model = fit_strategy(name, X, y)
    save_model(model, SAVED_MODELS_DIR / filename)
    print(f"Saved '{filename}'")
