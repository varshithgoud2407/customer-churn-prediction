import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model_5feat.pkl"

# ✅ Only 5 features (simple UI)
FEATURES = ["tenure", "Contract", "MonthlyCharges", "InternetService", "PaymentMethod"]

def main():
    df = pd.read_csv(DATA_PATH)

    # target
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # keep only 5 features
    X = df[FEATURES].copy()

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # preprocess
    cat_cols = ["Contract", "InternetService", "PaymentMethod"]
    num_cols = ["tenure", "MonthlyCharges"]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # model (good baseline + interpretable)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ])

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Saved 5-feature model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
