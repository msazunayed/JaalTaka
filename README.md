# JaalTaka

Minimal web app to classify banknotes as real or fake using a pre-trained CNN.

Usage
- Install: `pip install -r requirements.txt`
- Run: `uvicorn main:app --reload` and open http://127.0.0.1:8000

Files
- `main.py` — FastAPI app
- `model/` — model code and `saved/best_model.pth`
- `templates/index.html` — frontend

License: educational / research use.