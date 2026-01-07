import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
import joblib

class KNNGuardrail:
    def __init__(self, model_name='all-MiniLM-L6-v2', n_neighbors=5):
        # Используем легкую модель, чтобы не нагружать 1080 Ti
        self.encoder = SentenceTransformer(model_name)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        self.is_fitted = False

    def fit(self, texts, labels):
        print(f"Извлечение эмбеддингов для {len(texts)} примеров...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        self.knn.fit(embeddings, labels)
        self.is_fitted = True
        print("KNN Guardrail обучен.")

    def predict_proba(self, texts):
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        return self.knn.predict_proba(embeddings)[:, 1]

    def save(self, path):
        joblib.dump({"knn": self.knn, "encoder_name": 'all-MiniLM-L6-v2'}, path)

    def load(self, path):
        data = joblib.load(path)
        self.knn = data["knn"]
        self.is_fitted = True