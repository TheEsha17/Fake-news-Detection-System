import json, pathlib, math, joblib
from flask import Flask, jsonify, request, send_from_directory
from scipy.sparse import hstack

BASE = pathlib.Path(__file__).parent
FRONT = BASE.parent / "frontend"
MODEL_DIR = BASE / "model"
app = Flask(__name__, static_folder=None)

def load():
    try:
        m = joblib.load(MODEL_DIR / "model.joblib"); v = joblib.load(MODEL_DIR / "vectorizer.joblib")
        meta = json.load(open(MODEL_DIR / "metrics.json")) if (MODEL_DIR / "metrics.json").exists() else {}
        lab = json.load(open(MODEL_DIR / "labels.json")) if (MODEL_DIR / "labels.json").exists() else {"0": "REAL", "1": "FAKE"}
        return m, v, meta, lab
    except FileNotFoundError:
        return None, None, {}, {}
model, vectorizer, meta, labels = load()

@app.get("/api/health")
def health():
    return jsonify(model_loaded=model is not None)

@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify(error="No trained model found. Run `python train.py` (see README)."), 503
    d = request.get_json(silent=True) or {}
    title, text = (d.get("title") or "").strip(), (d.get("text") or "").strip()
    if len(title) < 5 or len(text.split()) < 30:
        return jsonify(error="Provide a headline (5+ characters) and an article of at least 30 words."), 400
    X = hstack([vectorizer["title"].transform([title]), vectorizer["text"].transform([text])]).tocsr()
    best = next((m for m in meta.get("models", []) if m["name"] == meta.get("best_model")), {})
    pred = int(model.predict(X)[0]); conf = None
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(X)[0][pred])
    return jsonify(label=labels[str(pred)], confidence=conf,
                   model=meta.get("best_model", type(model).__name__), confidence_available=conf is not None,
                   accuracy=best.get("accuracy"), precision=best.get("precision"))

@app.get("/api/model-metrics")
def metrics():
    if not meta:
        return jsonify(error="metrics.json not found. Run `python train.py`."), 503
    return jsonify(meta)

@app.get("/")
def home(): return send_from_directory(FRONT, "index.html")
@app.get("/<path:p>")
def static_files(p):
    if not (FRONT / p).is_file(): return jsonify(error="Not found"), 404
    return send_from_directory(FRONT, p)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
