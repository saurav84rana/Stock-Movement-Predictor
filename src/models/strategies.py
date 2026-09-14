import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

class ModelStrategy(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ModelStrategy":
        """Train the model, with `X` dataset, and `y` targets"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns target for each row"""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns probability that target = 1, for each row"""
        pass


class LogisticRegressionStrategy(ModelStrategy):
    """Logistic Regression, with Scaler to normalise dataset beforehand"""
    name = "logistic_regression"

    def __init__(self, random_state: int = 42, **kwargs):
        # logistical regression's loss function is sensitive to feature's scale
        # so, it needs to be normalised
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=random_state, **kwargs))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]

        
class GradientBoostingStrategy(ModelStrategy):
    """Gradient Boosting Classifier"""
    name = "gradient_boosting"

    def __init__(self, random_state: int = 42, **kwargs):
        self.model = HistGradientBoostingClassifier(random_state=random_state, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]


def get_strategy(name: str, random_state: int = 42, **kwargs) -> ModelStrategy:
    """Factory: returns `name` strategy, initialised with random state `random_state`

    Strategies options are: ['logistic_regression', 'gradient_boosting']"""

    strategies: dict[str, type[ModelStrategy]] = {
        "logistic_regression": LogisticRegressionStrategy,
        "gradient_boosting": GradientBoostingStrategy
    }

    if name not in strategies:
        raise ValueError(
            f"Unknown strategy: '{name}'. Strategies options are: {list(strategies)}"
        )
    
    return strategies[name](random_state=random_state, **kwargs)
