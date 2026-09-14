import pandas as pd
import numpy as np
import joblib
from src.models.strategies import get_strategy, ModelStrategy

def fit_strategy(name: str, X_train: pd.DataFrame, y_train: np.ndarray, **params) -> ModelStrategy:
    """Fit a model `name` on given data `X_train` and `y_train`"""
    model = get_strategy(name, **params)
    model.fit(X_train, y_train)
    return model

def save_model(model: ModelStrategy, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str) -> ModelStrategy:
    return joblib.load(path)