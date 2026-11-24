"""Rebuilds the InFact Colab notebook (WELFake dataset), evaluates six models, saves the Linear SVM.
Usage:  python train.py [path/to/WELFake_Dataset.csv] [output_dir]
"""
import sys, json, pathlib, joblib, numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/WELFake_Dataset.csv"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "model"); OUT.mkdir(exist_ok=True)
DEPLOY = "Linear SVM"
# In WELFake the sample rows in the notebook show label 1 = FAKE (e.g. hyperpartisan "[VIDEO]" posts)
# and label 0 = REAL. If your own check says the opposite, swap these two values.
LABELS = {"0": "REAL", "1": "FAKE"}

df = pd.read_csv(CSV)
df = df[df["text"].fillna("").astype(str).str.strip().str.len() >= 20].drop_duplicates(subset="text").copy()
df["title"] = df["title"].fillna("")
X, y = df[["title", "text"]], df["label"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
assert not set(Xtr.text.str.strip()) & set(Xte.text.str.strip()), "train/test overlap"

tv = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
xv = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
Atr = hstack([tv.fit_transform(Xtr.title), xv.fit_transform(Xtr.text)]).tocsr()
Ate = hstack([tv.transform(Xte.title), xv.transform(Xte.text)]).tocsr()

preds, fitted = {}, {}
def run(name, model, a, b, yy=ytr):
    print("Training", name); model.fit(a, yy); preds[name] = model.predict(b); fitted[name] = model

run("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), Atr, Ate)
run("Linear SVM", LinearSVC(C=1.0, random_state=42, max_iter=2000), Atr, Ate)
run("Naive Bayes", MultinomialNB(), Atr, Ate)
svd = TruncatedSVD(n_components=50, random_state=42)
Str, Ste = svd.fit_transform(Atr), svd.transform(Ate)
idx = np.random.RandomState(42).choice(Str.shape[0], min(15000, Str.shape[0]), replace=False)
run("KNN", KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute", n_jobs=-1), Str[idx], Ste, ytr.iloc[idx])
sel = SelectKBest(chi2, k=min(3000, Atr.shape[1])); Dtr, Dte = sel.fit_transform(Atr, ytr), sel.transform(Ate)
g = GridSearchCV(DecisionTreeClassifier(random_state=42), {"max_depth": [10, 15, 20], "min_samples_leaf": [5, 10]}, cv=3, n_jobs=-1)
run("Decision Tree", g, Dtr, Dte)
run("Random Forest", RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42), Dtr, Dte)

res = [{"name": n, "accuracy": accuracy_score(yte, p), "precision": precision_score(yte, p, average="weighted"),
        "recall": recall_score(yte, p, average="weighted"), "f1": f1_score(yte, p, average="weighted"),
        "confusion_matrix": confusion_matrix(yte, p, labels=[0, 1]).tolist()} for n, p in preds.items()]
joblib.dump(fitted[DEPLOY], OUT / "model.joblib")
joblib.dump({"title": tv, "text": xv}, OUT / "vectorizer.joblib")
json.dump(LABELS, open(OUT / "labels.json", "w"))
json.dump({"best_model": DEPLOY, "models": res, "averaging": "weighted", "class_order": [LABELS["0"], LABELS["1"]],
           "dataset": {"train_size": len(Xtr), "test_size": len(Xte)}}, open(OUT / "metrics.json", "w"), indent=2)
for r in res: print(f'{r["name"]:20s} acc={r["accuracy"]:.4f} f1={r["f1"]:.4f}')
print("Saved", DEPLOY, "to", OUT)
