# customer-churn-prediction

# customer-churn-prediction

This repository contains an end-to-end demo for predicting customer churn using the Telco Customer Churn dataset. It includes data, notebooks, a small 5-feature training script, a FastAPI backend, and a Streamlit frontend. The project is set up so both backend and frontend install dependencies from a single `requirements.txt` located at the repository root.

**What changed (summary)**
- Unified dependency management: both services use the root `requirements.txt`.
- Lightweight 5-feature model: `src/train_simple.py` trains and saves `models/churn_model_5feat.pkl` (smaller and faster for demos).
- Backend updated: `backend/app.py` now loads the 5-feature model and exposes `/health` and `/predict`.
- Frontend updated: `frontend/app.py` is a Streamlit UI that calls the backend `/predict` endpoint.
- Git cleanup: `.venv/` has been added to `.gitignore` and untracked from the repository.

## Project layout

See the active files and folders of interest:

```
customer-churn-prediction/
├── backend/                 # FastAPI backend service
│   └── app.py
├── frontend/                # Streamlit frontend
│   └── app.py
├── data/                    # Raw dataset CSV
├── models/                  # Trained model artifacts
│   └── churn_model_5feat.pkl
├── notebooks/               # EDA and experiments
├── src/                     # training & helper scripts
│   └── train_simple.py      # trains 5-feature model
├── requirements.txt         # Shared dependencies for both services
├── docker-compose.yml       # Builds backend & frontend from repo root
├── backend/Dockerfile
├── frontend/Dockerfile
└── README.md
```

## Quickstart — local (recommended for development)

1) Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

2) Install dependencies from the shared `requirements.txt`:

```powershell
pip install -r requirements.txt
```

3) Train the 5-feature model and save it to `models/` (required before starting the backend):

```powershell
python src/train_simple.py
# This will create models/churn_model_5feat.pkl
```

4) Start the FastAPI backend (runs on port 8000):

```powershell
uvicorn backend.app:app --reload --port 8000
```

Verify health:

```powershell
curl http://127.0.0.1:8000/health
# Expected: JSON or plain text status confirming model loaded
```

5) Start the Streamlit frontend (runs on port 8501):

```powershell
streamlit run frontend/app.py
```

The Streamlit UI posts to the backend `/predict` endpoint to get churn probability.

## Quickstart — Docker (builds from repo root)

Both services are configured to use the repository root as the Docker build context so they install the single root `requirements.txt`.

Bring up both services with compose:

```powershell
docker-compose up --build
```

This will build images using `backend/Dockerfile` and `frontend/Dockerfile` and run the services together.

## Details & troubleshooting

- Requirements: The top-level `requirements.txt` contains runtime dependencies (pandas, scikit-learn, fastapi, uvicorn, streamlit, joblib, requests, etc.). We removed the heavy `jupyter` meta-package from the shared requirements to avoid obscure build failures. If you need Jupyter for experimentation, install it in your dev env separately:

```powershell
pip install jupyterlab
```

- Model artifact: The training script `src/train_simple.py` trains a compact logistic-regression pipeline using five engineered features and saves the artifact as `models/churn_model_5feat.pkl`. The backend expects that file to exist.

- Backend server exit behavior: if `uvicorn` starts and then immediately shuts down, ensure:
	- You ran `python -m venv .venv` and activated it prior to starting `uvicorn` so the installed `uvicorn` binary runs from the same environment; or run with the interpreter directly: `python -m uvicorn backend.app:app --reload --port 8000`.
	- If using an editor/runner that spawns processes, run `uvicorn` from a terminal so the process stays attached.

- Git cleanup: `.venv/` has been appended to `.gitignore` and the venv was untracked with `git rm -r --cached .venv`. If you still see large files on the remote and want to remove them from history, consider using `git filter-repo` or the GitHub UI guidance for removing large files.

## Useful file references

- Training script: [src/train_simple.py](src/train_simple.py)
- Backend entrypoint: [backend/app.py](backend/app.py)
- Frontend (Streamlit): [frontend/app.py](frontend/app.py)
- Saved model: [models/churn_model_5feat.pkl](models/churn_model_5feat.pkl)

## Next steps (suggested)

- If you want, I can:
	- Run the backend locally and debug why `uvicorn` is exiting immediately.
	- Push a small convenience script that starts backend + frontend with proper environment activation.

Thanks — open an issue or ask here for the next action you want me to take.
