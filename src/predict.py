"""Simple prediction helper for the Customer Churn project.

This file is intentionally small — keep production logic here.
"""

import joblib
import pandas as pd
from typing import Union


def load_model(path: str):
    """Load a model saved with joblib."""
    return joblib.load(path)


def predict(model, X: Union[pd.DataFrame, list]):
    """Return predicted probabilities (or classes) for input X.

    - If `model` has `predict_proba`, returns probability of positive class.
    - Accepts a list (converted to DataFrame) or a DataFrame.
    """
    if isinstance(X, list):
        X = pd.DataFrame(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


if __name__ == "__main__":
    print("Run `from src.predict import load_model, predict` in your code.")
