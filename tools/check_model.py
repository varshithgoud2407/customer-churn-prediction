from pathlib import Path
p = Path(__file__).resolve().parent.parent / 'models' / 'churn_pipeline.joblib'
print('script path:', Path(__file__).resolve())
print('model path:', p)
print('exists:', p.exists())
try:
    print('size:', p.stat().st_size)
except Exception as e:
    print('size error:', e)
