# mvp_automl.py
import os
import json
import pickle
from itertools import product
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm


# ------------------------------
# Feature extraction utilities
# ------------------------------

def compute_centroids(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute centroids for binary classification (Allowed vs Attack)."""
    centroids = {}
    for label in np.unique(y):
        centroids[label] = X[y == label].mean(axis=0)
    return centroids


def centroid_margin_feature(X: np.ndarray, centroids: Dict[str, np.ndarray]) -> np.ndarray:
    """Distance to centroids as feature."""
    allowed_centroid = centroids[0]
    attack_centroid = centroids[1]
    dist_to_allowed = np.linalg.norm(X - allowed_centroid, axis=1)
    dist_to_attack = np.linalg.norm(X - attack_centroid, axis=1)
    margin = dist_to_allowed - dist_to_attack
    return margin.reshape(-1, 1)


# ------------------------------
# Pipeline
# ------------------------------

class MVP_Pipeline:
    def __init__(
        self,
        feature_set: List[str],
        stage1_model_type: str,
        stage1_threshold: float,
        stage2_filter_type: Optional[str] = None,
        stage2_threshold: Optional[float] = None
    ):
        self.feature_set = feature_set
        self.stage1_model_type = stage1_model_type
        self.stage1_threshold = stage1_threshold
        self.stage2_filter_type = stage2_filter_type
        self.stage2_threshold = stage2_threshold

        self.stage1_model = None
        self.stage2_model = None
        self.scaler = None
        self.centroids = None
        self.isolation_model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        # ----------------
        # Feature selection
        # ----------------
        features = []
        if 'embedding' in self.feature_set:
            features.append(np.vstack(X['embedding'].values))
        if 'length_features' in self.feature_set:
            length_feats = X['text'].apply(lambda t: [len(t.split()), len(t)])
            features.append(np.vstack(length_feats.values))
        if 'centroid_margin' in self.feature_set:
            # compute centroids on embeddings
            emb = np.vstack(X['embedding'].values)
            self.centroids = compute_centroids(emb, y)
            margin_feat = centroid_margin_feature(emb, self.centroids)
            features.append(margin_feat)
        if 'anomaly_score' in self.feature_set:
            emb = np.vstack(X['embedding'].values)
            self.isolation_model = IsolationForest(random_state=42)
            self.isolation_model.fit(emb)
            iso_score = -self.isolation_model.score_samples(emb).reshape(-1, 1)
            features.append(iso_score)
        if 'keyword_flags' in self.feature_set:
            key_words = ['exploit', 'bypass', 'hack', 'malware']
            kw_flags = X['text'].apply(lambda t: [int(k in t.lower()) for k in key_words])
            features.append(np.vstack(kw_flags.values))
        if not features:
            raise ValueError("No features selected!")
        X_feat = np.hstack(features)

        # ----------------
        # Scale features
        # ----------------
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_feat)

        # ----------------
        # Stage 1: recall-heavy model
        # ----------------
        if self.stage1_model_type == 'logreg':
            self.stage1_model = LogisticRegression(class_weight='balanced', max_iter=500)
            self.stage1_model.fit(X_scaled, y)
        elif self.stage1_model_type == 'svm':
            self.stage1_model = LinearSVC(class_weight='balanced', max_iter=500)
            self.stage1_model.fit(X_scaled, y)
        elif self.stage1_model_type == 'catboost':
            from catboost import CatBoostClassifier
            self.stage1_model = CatBoostClassifier(
                iterations=200, depth=6, verbose=0, class_weights=[1,1]
            )
            self.stage1_model.fit(X_scaled, y)
        else:
            raise ValueError(f"Unknown Stage1 model type: {self.stage1_model_type}")

        # ----------------
        # Stage 2: optional FPR filter
        # ----------------
        if self.stage2_filter_type == 'centroid_distance' and self.centroids is None:
            raise ValueError("Centroids required for centroid_distance filter.")
        if self.stage2_filter_type == 'isolation':
            # already fitted
            if self.isolation_model is None:
                raise ValueError("Isolation model required for anomaly filter.")

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        features = []
        if 'embedding' in self.feature_set:
            features.append(np.vstack(X['embedding'].values))
        if 'length_features' in self.feature_set:
            length_feats = X['text'].apply(lambda t: [len(t.split()), len(t)])
            features.append(np.vstack(length_feats.values))
        if 'centroid_margin' in self.feature_set:
            emb = np.vstack(X['embedding'].values)
            margin_feat = centroid_margin_feature(emb, self.centroids)
            features.append(margin_feat)
        if 'anomaly_score' in self.feature_set:
            emb = np.vstack(X['embedding'].values)
            iso_score = -self.isolation_model.score_samples(emb).reshape(-1, 1)
            features.append(iso_score)
        if 'keyword_flags' in self.feature_set:
            key_words = ['exploit', 'bypass', 'hack', 'malware']
            kw_flags = X['text'].apply(lambda t: [int(k in t.lower()) for k in key_words])
            features.append(np.vstack(kw_flags.values))
        if not features:
            raise ValueError("No features selected!")
        X_feat = np.hstack(features)
        X_scaled = self.scaler.transform(X_feat)
        return X_scaled

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.transform_features(X)

        # Stage 1 score
        if hasattr(self.stage1_model, "predict_proba"):
            score = self.stage1_model.predict_proba(X_scaled)[:, 1]
        else:
            # SVM decision function
            score = self.stage1_model.decision_function(X_scaled)

        stage1_pred = (score >= self.stage1_threshold).astype(int)

        # Stage 2 filter
        if self.stage2_filter_type == 'centroid_distance':
            emb = np.vstack(X['embedding'].values)
            margin_feat = centroid_margin_feature(emb, self.centroids).flatten()
            stage2_mask = margin_feat >= self.stage2_threshold
            final_pred = stage1_pred & stage2_mask
        elif self.stage2_filter_type == 'isolation':
            emb = np.vstack(X['embedding'].values)
            iso_score = -self.isolation_model.score_samples(emb)
            stage2_mask = iso_score <= self.stage2_threshold
            final_pred = stage1_pred & stage2_mask
        else:
            final_pred = stage1_pred

        return final_pred


# ------------------------------
# AutoML search
# ------------------------------

def automl_search(X: pd.DataFrame, y: np.ndarray, verbose=True) -> MVP_Pipeline:
    FEATURE_SETS = [
        ['embedding'],
        ['embedding', 'centroid_margin'],
        ['embedding', 'anomaly_score'],
        ['embedding', 'length_features', 'keyword_flags'],
        ['embedding', 'centroid_margin', 'anomaly_score']
    ]
    STAGE1_MODELS = ['logreg', 'svm', 'catboost']  # catboost optional if installed
    STAGE1_THRESHOLDS = [0.3, 0.5, 0.7]
    STAGE2_FILTERS = [None, 'centroid_distance', 'isolation']
    STAGE2_THRESHOLDS = [0.3, 0.5, 0.7]

    best_score = -1
    best_pipeline = None

    for features, s1_model, s1_thresh, s2_filter, s2_thresh in tqdm(
        product(FEATURE_SETS, STAGE1_MODELS, STAGE1_THRESHOLDS, STAGE2_FILTERS, STAGE2_THRESHOLDS),
        total=len(FEATURE_SETS) * len(STAGE1_MODELS) * len(STAGE1_THRESHOLDS) * len(STAGE2_FILTERS) * len(STAGE2_THRESHOLDS),
        desc="AutoML search"
    ):
        try:
            pipe = MVP_Pipeline(features, s1_model, s1_thresh, s2_filter, s2_thresh)
            pipe.fit(X, y)
            y_pred = pipe.predict(X)
            # FPR check
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            recall_all = recall_score(y, y_pred)
            # score penalizes if FPR>0.05
            score = recall_all if fpr <= 0.05 else -1

            if score > best_score:
                best_score = score
                best_pipeline = pipe
        except Exception:
            continue

    if verbose:
        print(f"Best AutoML score: {best_score}")
        print(f"Best features: {best_pipeline.feature_set}")
        print(f"Stage1: {best_pipeline.stage1_model_type}, threshold={best_pipeline.stage1_threshold}")
        print(f"Stage2: {best_pipeline.stage2_filter_type}, threshold={best_pipeline.stage2_threshold}")

    # Save pipeline + artifacts
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/best_pipeline.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    return best_pipeline


# ------------------------------
# Example usage
# ------------------------------

if __name__ == "__main__":
    # dummy dataframe: text + embeddings + label
    df = pd.read_pickle("train_embeddings.pkl")  # columns: ['text', 'embedding', 'label']
    X = df[['text', 'embedding']]
    y = df['label'].values

    best_pipe = automl_search(X, y)
