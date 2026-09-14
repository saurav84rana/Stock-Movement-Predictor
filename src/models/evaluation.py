import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from src.models.strategies import LogisticRegressionStrategy, GradientBoostingStrategy, get_strategy
from src.validation.baselines import always_up, buy_and_hold
from src.validation.splits import walk_forward_splits

MODELS = ["logistic_regression", "gradient_boosting"]
NON_FEATURE_COLS = ["Date", "Open", "High", "Low", "Close", "Volume", "ticker", "target"]

# -- helper functions --
def _feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]

def summarise_by_column(df: pd.DataFrame, col: str = "model") -> pd.DataFrame:
    """Aggregate `df`'s per fold metrics into mean/std by column `col`, 
    automatically see if n_iter exists, and will make a list for it accordingly"""

    aggregations = { 
        "accuracy_mean": ("accuracy", "mean"), "accuracy_std": ("accuracy", "std"),
        "roc_auc_mean": ("roc_auc", "mean"), "roc_auc_std": ("roc_auc", "std"),
        "precision_mean": ("precision", "mean"), "precision_std": ("precision", "std"),
        "recall_mean": ("recall", "mean"), "recall_std": ("recall", "std"),
        "f1_mean": ("f1_score", "mean"), "f1_std": ("f1_score", "std"),
    }

    if "n_iter" in df.columns:
        aggregations["n_iter"] = ("n_iter", list)

    summary = (
        df
        .groupby(col)
        .agg(**aggregations)
        .reset_index()
    )

    return summary


def trim_tail(df: pd.DataFrame, gap: int) -> pd.DataFrame:
    """Drop the last `gap` trading days from `df`"""
    if gap <= 0:
        return df
    unique_dates = np.sort(df["Date"].unique())
    cutoff_date = unique_dates[-gap - 1]
    return df[df["Date"] <= cutoff_date]


# -- evaluation functions --
def parameter_sweep(dev_df: pd.DataFrame, model_name: str, param_grid: list[dict], gap: int = 25
                    ) -> pd.DataFrame:
    """Conduct walk forward cross validation for one model `model_name`, 
    across a grid of hyperparameters `param_grid`,
    and returns a per fold result DataFrame

    (Used to view models and models' hyperparameter performance)
    """
    results: list[dict] = []
    feature_cols = _feature_cols(dev_df)

    for params in param_grid:
        model_label = f"{model_name} (" + ", ".join(f"{k}={v}" for k, v in sorted(params.items())) + ")"
        # e.g. "gradient_boosting (learning_rate=1, max_iter=1000)"

        for fold, (train_idx, test_idx) in enumerate(walk_forward_splits(dev_df, gap=gap)):
            X_train = dev_df[feature_cols].iloc[train_idx]
            y_train = dev_df["target"].iloc[train_idx]

            X_test = dev_df[feature_cols].iloc[test_idx]
            y_test = dev_df["target"].iloc[test_idx]

            model = get_strategy(model_name, **params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            n_iter = -1
            if model_name == "gradient_boosting":
                n_iter = (model.model.n_iter_)
            elif model_name == "logistic_regression":
                n_iter = (model.model.named_steps["classifier"].n_iter_[0])

            results.append({
                "model": model_label,
                "fold": fold,
                "n_iter": n_iter,
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            })

    return pd.DataFrame(results)


def models_baseline_validation(dev_df: pd.DataFrame, gap: int = 25):
    """Perform cross validation walk forward split of training df `dev_df` for ML models 
    ("logistic_regression", "gradient_boosting), as well as baseline ("always up")
    
    (Used to evaluate model)
    """
    results: list[dict] = []
    feature_cols = _feature_cols(dev_df)

    for fold, (train_idx, test_idx) in enumerate(walk_forward_splits(dev_df, gap=gap)):
        X_train = dev_df[feature_cols].iloc[train_idx]
        y_train = dev_df["target"].iloc[train_idx]

        X_test = dev_df[feature_cols].iloc[test_idx]
        y_test = dev_df["target"].iloc[test_idx]

        # ML models
        for model_name in MODELS:
            # get model
            model = get_strategy(model_name)

            # train model
            model.fit(X_train, y_train)

            # get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            results.append({
                "model": model_name,
                "fold": fold,
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            })

        # Always up baseline
        baseline_predict = always_up(len(y_test))

        results.append({
            "model": "always_up",
            "fold": fold,
            "accuracy": accuracy_score(y_test, baseline_predict),
            "roc_auc": roc_auc_score(y_test, baseline_predict),  # prob of predicting class 1 is 1 anyway
            "precision": precision_score(y_test, baseline_predict, zero_division=0),
            "recall": recall_score(y_test, baseline_predict, zero_division=0),
            "f1_score": f1_score(y_test, baseline_predict, zero_division=0)
        })

    return pd.DataFrame(results)


def final_holdout_eval(dev_df: pd.DataFrame, test_df:pd.DataFrame, 
                       model_names=("logistic_regression", "gradient_boosting")
                      ) -> tuple[pd.DataFrame, float]:
    """Train on each model in `model_names` on training df `dev_df`, 
    test on held out df `test_df`. Returns results DataFrame and buy and hold results
    
    Please ensure `dev_df` and `test_df` is appropriately spaced with gap
    
    (Used to get final verdict on chosen model)"""
    results: list[dict] = []
    feature_cols = _feature_cols(dev_df)

    X_train, y_train = dev_df[feature_cols], dev_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    # model training and testing
    for model_name in model_names:
        model = get_strategy(model_name)
        model.fit(X_train, y_train)
        model_pred = model.predict(X_test)
        model_pred_proba = model.predict_proba(X_test)

        results.append({
            "model": model_name,
            "accuracy": accuracy_score(y_test, model_pred),
            "roc_auc": roc_auc_score(y_test, model_pred_proba),
            "precision": precision_score(y_test, model_pred),
            "recall": recall_score(y_test, model_pred),
            "f1_score": f1_score(y_test, model_pred)
        })

    # Always up baseline
    baseline_predict = always_up(len(y_test))

    results.append({
        "model": "always_up",
        "accuracy": accuracy_score(y_test, baseline_predict),
        "roc_auc": roc_auc_score(y_test, baseline_predict),  # prob of predicting class 1 is 1 anyway
        "precision": precision_score(y_test, baseline_predict, zero_division=0),
        "recall": recall_score(y_test, baseline_predict, zero_division=0),
        "f1_score": f1_score(y_test, baseline_predict, zero_division=0),
    })

    results_df = pd.DataFrame(results)
    bh_return = buy_and_hold(test_df, test_df["Date"].min(), test_df["Date"].max())
    return results_df, bh_return


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).parent.parent.parent
    RESULTS_RAW_DIR = PROJECT_DIR / "results" / "raw"
    RESULTS_PROCESSED_DIR = PROJECT_DIR / "results" / "processed"
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

    # make sure destinations actually exist before saving to it
    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


    df = pd.read_csv(PROCESSED_DIR / "processed.csv", parse_dates=["Date"]
                     ).reset_index(drop=True)

    # hard cut off date to hold out any date after 2025-01-01 for test
    # additionally add a purge gap from dev_df
    cutoff = pd.to_datetime("2025-01-01")
    dev_df, test_df = df[df["Date"] < cutoff], df[df["Date"] >= cutoff]
    dev_df = trim_tail(dev_df, gap=25)

    # raw reesults
    results_val = models_baseline_validation(dev_df)
    results_val.to_csv(RESULTS_RAW_DIR / "validation_eval.csv", index=False)

    results_final, bh = final_holdout_eval(dev_df, test_df)
    results_final.to_csv(RESULTS_RAW_DIR / "test_eval.csv", index=False)

    # summarised (procesed) results
    summarise_by_column(results_val).to_csv(
        RESULTS_PROCESSED_DIR / "validation_eval_summary.csv", index=False)