import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

MODEL_PATH = "../models/churn_pipeline.joblib"
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_data():
    return pd.read_csv(DATA_PATH)

def evaluate():
    model = joblib.load(MODEL_PATH)
    df = load_data()
    df = df.select_dtypes(include=["number"]).dropna()
    if "Churn" in df.columns:
        y = df["Churn"].astype(int)
        X = df.drop(columns=["Churn"])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate()
