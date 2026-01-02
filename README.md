# customer-churn-prediction

## 📌 Problem Statement
Predict customer churn to help businesses take proactive retention actions.

## 🔍 Dataset
Telco Customer Churn Dataset (IBM / Kaggle) — the CSV lives at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## ☁️ Google Colab
You can run an interactive version of this project on Google Colab:

https://colab.research.google.com/drive/10BWhReEENwo-IJqIhZJRzWIyAl9DA1hG?usp=sharing

## 🔧 Project Structure

```
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   └── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run locally

### 1) Train + save model
Run the notebook or training script so `models/churn_pipeline.joblib` exists.

### 2) Start FastAPI backend
```bash
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

### Using Docker (recommended)
Both the backend and frontend use the single `requirements.txt` at the repository root. The compose file builds with the repository root as the build context and points each service at its service Dockerfile so both services install the same dependencies.

To build and run both services with Docker Compose:
```powershell
docker-compose up --build
```

To build images individually (optional):
```powershell
docker build -f backend/Dockerfile -t churn-backend .
docker build -f frontend/Dockerfile -t churn-frontend .
```

## ✅ Tips
- Keep raw data in `data/` and never commit large model artifacts.
- Use `src/` for production code and small utilities.

## 📁 Repository
Name the GitHub repo: `customer-churn-prediction` (public). Do not initialize with README on GitHub if you push an existing repo.
