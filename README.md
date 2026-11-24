# InFact — Facts over fiction

Flask + scikit-learn backend with a single-page frontend, both served by one command.

## Run
```
cd backend
pip install -r requirements.txt     # scikit-learn must be 1.9.1 to load the saved model
python app.py                       # open http://localhost:5000
```

## What is connected
- `backend/model/model.joblib` — Linear SVM from the Colab notebook (best of six on accuracy).
- `backend/model/vectorizer.joblib` — title and text TF-IDF vectorizers.
- `backend/model/labels.json` — class 0 = REAL, 1 = FAKE (confirmed from the model's strongest words).
- `backend/model/metrics.json` — scores copied from the notebook's printed results, served at `GET /api/model-metrics`.
- `POST /api/predict` takes `{"title": "...", "text": "..."}` and returns `label`, `model`, and `confidence` (null for Linear SVM, which has no probability).

`python train.py` can rebuild everything from `WELFake_Dataset.csv` if you want to retrain.

## Limits worth knowing
- The model learned dataset habits (words like "Reuters", "said", "featured image"), not facts. Treat results as one signal.
- The notebook's scores are probably optimistic: its split let some identical articles appear in both training and test data.
