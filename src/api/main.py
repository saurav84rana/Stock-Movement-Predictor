from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from pandas.tseries.offsets import BDay
import pandas as pd
from src.models.train import load_model

app = FastAPI(title="Stock Movement Predictor")

DISCLAIMER = """
Neither Logistic Regression nor Gradient Boosting beats Always Up baseline on held-out 2025 data. 
Please see README.md and notebooks/notes/modelling_notes.md for detailed information and analysis.
""".strip()

# Global path variables
PROJECT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"

# load df once
processed_df = pd.read_csv(PROCESSED_DIR / "processed.csv", parse_dates=["Date"])
test_eval_df = pd.read_csv(RESULTS_DIR / "raw" / "test_eval.csv")

# models and features
MODELS = {
    "logistic_regression": load_model(SAVED_MODELS_DIR / "production_logreg.joblib"),
    "gradient_boosting": load_model(SAVED_MODELS_DIR / "production_gbm.joblib")
}

NON_FEATURE_COLS = ["Date", "Open", "High", "Low", "Close", "Volume", "ticker", "target"]
FEATURE_COLS = [c for c in processed_df.columns if c not in NON_FEATURE_COLS]

# -- FastAPI --
class PredictionResponse(BaseModel):
    ticker: str
    prediction_for_date: str
    model: str
    prediction: str
    probability: float
    disclaimer: str = DISCLAIMER

@app.get("/predict", response_model=PredictionResponse)
def predict(ticker: str, model_name: str = "logistic_regression"):
    if model_name not in MODELS:
        raise HTTPException(400, f"Unknown model: {model_name}")

    rows = processed_df[processed_df["ticker"] == ticker]
    if rows.empty:
        raise HTTPException(404, f"No data for ticker: {ticker}")

    latest_row = rows.sort_values("Date").iloc[[-1]]
    y_predict = latest_row[FEATURE_COLS]

    model = MODELS[model_name]

    pred = "up" if model.predict(y_predict)[0] == 1 else "down"

    proba_up = model.predict_proba(y_predict)[0]
    probability = proba_up if pred == "up" else 1 - proba_up

    return {
        "ticker": ticker,
        "prediction_for_date": str((latest_row["Date"].iloc[0] + BDay(1)).date()),
        "model": model_name,
        "prediction": pred,
        "probability": probability,
        "disclaimer": DISCLAIMER
    }

@app.get("/model-info")
def model_info():
    with open(RESULTS_DIR / "raw" / "test_extra.txt", "r") as f:
        bh_return = float(f.readline().split(": ")[-1])
    return {
        "evaluation_results": test_eval_df.to_dict(orient="records"),
        "buy_and_hold_return": bh_return,
        "disclaimer": DISCLAIMER
    }
