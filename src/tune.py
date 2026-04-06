import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"

def load_data():
    df = pd.read_csv(DATA_PATH)
    # drop identifier if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    # map target to 0/1
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def build_pipeline(df):
    # split features by dtype
    y = df["Churn"]
    X = df.drop(columns=["Churn"]) if "Churn" in df.columns else df.iloc[:, :-1]

    # simple column selection
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=[object, "category"]).columns.tolist()

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
    ], remainder="drop")

    # use imblearn pipeline to allow SMOTE
    pipe = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1))
    ])

    return X, y, pipe


def tune():
    df = load_data()
    X, y, pipe = build_pipeline(df)

    # compute reasonable scale_pos_weight for XGBoost
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = max(1.0, neg / max(1.0, pos))

    param_distributions = {
        "model__n_estimators": [50, 100, 200, 400],
        "model__max_depth": [3, 4, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.5, 0.7, 1.0],
        "model__scale_pos_weight": [scale_pos_weight]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        verbose=2,
        random_state=42,
    )

    print("Starting hyperparameter search (this may take several minutes)...")
    search.fit(X, y)

    print("Best ROC-AUC:", search.best_score_)
    print("Best params:", search.best_params_)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_PATH)
    print(f"✅ Saved best pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    tune()
