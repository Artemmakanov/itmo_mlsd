from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer

class KNNBaselineGR:
    def __init__(self, k=5, model_name="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")

    def fit(self, texts, y):
        X = self.encoder.encode(texts, show_progress_bar=False)
        self.clf.fit(X, y)

    def predict(self, texts):
        X = self.encoder.encode(texts, show_progress_bar=False)
        return self.clf.predict(X)

    def predict_proba(self, texts):
        X = self.encoder.encode(texts, show_progress_bar=False)
        return self.clf.predict_proba(X)[:, 1]
