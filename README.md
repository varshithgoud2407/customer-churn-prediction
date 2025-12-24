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

## ▶️ Run locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

Open the notebook:

```powershell
cd 'C:\Users\varsh\Downloads\customer-churn-prediction'
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

## ✅ Tips
- Keep raw data in `data/` and never commit large model artifacts.
- Use `src/` for production code and small utilities.

## 📁 Repository
Name the GitHub repo: `customer-churn-prediction` (public). Do not initialize with README on GitHub if you push an existing repo.
