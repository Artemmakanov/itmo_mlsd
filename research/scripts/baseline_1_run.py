import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sentence_transformers import SentenceTransformer
import torch

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------
# Эмбеддинги через MiniLM
# --------------------------
class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.encoder.eval()

    def encode(self, texts):
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return embeddings
# --------------------------
# Загрузка и фильтр данных
# --------------------------
def load_data(filename):
    path_base = f"./data/{filename}.csv"
    path_labeled = f"./data/{filename}_labeled.csv"
    df_labeled = pd.read_csv(path_labeled)
    df_base = pd.read_csv(path_base)
    df_labeled = df_labeled[df_labeled['segment'].isin(["Blind Spot", "Naturally Allowed"])][['segment']]
    return df_base.merge(df_labeled, left_index=True, right_index=True, how='inner')

# --------------------------
# Поиск threshold для FPR < 5%
# --------------------------
def find_best_threshold(y_true, y_probs, max_fpr=0.05):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    thresholds = np.append(thresholds, 1.0)  # чтобы совпала длина
    best_idx = None
    best_recall = 0
    for i, thresh in enumerate(thresholds):
        y_pred = (y_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        if fpr <= max_fpr and recall[i] > best_recall:
            best_recall = recall[i]
            best_idx = i
    if best_idx is None:
        raise Exception(f"FPR always > {max_fpr}")
    return thresholds[best_idx]

def save_prediction_plot(y_true, y_probs, threshold, path):
    plt.figure(figsize=(8, 5))

    pos = y_probs[y_true == 1]
    neg = y_probs[y_true == 0]

    plt.hist(
        neg, bins=50, alpha=0.6, density=True,
        label="y=0 (Refusal)"
    )
    plt.hist(
        pos, bins=50, alpha=0.6, density=True,
        label="y=1 (Allowed)"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"threshold = {threshold:.3f}"
    )

    plt.xlabel("Predicted probability P(y=1)")
    plt.ylabel("Density")
    plt.title("Baseline_1 — Test predictions distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# --------------------------
# Grid search KNN
# --------------------------
def grid_search_knn(X_train, y_train, X_eval, y_eval, param_grid):
    best_score = -1
    best_params = None
    for params in ParameterGrid(param_grid):
        model = KNeighborsClassifier(**params)
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_eval)[:, 1]
        try:
            threshold = find_best_threshold(y_eval, y_probs)
        except:
            continue
        y_pred = (y_probs >= threshold).astype(int)
        recall = recall_score(y_eval, y_pred)
        if recall > best_score:
            best_score = recall
            best_params = {**params, "threshold": threshold, "recall": best_score}
        print(recall, params)
    return best_params

# --------------------------
# Основная функция
# --------------------------
def main():

    print("Loading data...")
    train_df = load_data('train')
    eval_df = load_data('eval')
    test_df = load_data('test')

    embedder = Embedder()

    print("Encoding train data...")
    X_train = embedder.encode(train_df['text'].tolist())
    y_train = train_df['label'].values

    print("Encoding eval data...")
    X_eval = embedder.encode(eval_df['text'].tolist())
    y_eval = eval_df['label'].values

    print("Encoding test data...")
    X_test = embedder.encode(test_df['text'].tolist())
    y_test = test_df['label'].values

    # --------------------------
    # Grid search KNN
    # --------------------------
    param_grid = {
        "n_neighbors": list(range(1, 10, 1)) + list(range(10, 200, 10)),
        "weights": ["uniform", "distance"],
        "metric": ["cosine", "euclidean"]
    }

    print("Searching best KNN hyperparameters on eval...")
    best_params = grid_search_knn(X_train, y_train, X_eval, y_eval, param_grid)

    # --------------------------
    # Обучаем финальную модель на train + eval
    # --------------------------
    X_final = np.vstack([X_train, X_eval])
    y_final = np.hstack([y_train, y_eval])

    final_model = KNeighborsClassifier(
        n_neighbors=best_params["n_neighbors"],
        weights=best_params["weights"],
        metric=best_params["metric"]
    )
    final_model.fit(X_final, y_final)

    # --------------------------
    # Предсказания на тесте
    # --------------------------
    test_probs = final_model.predict_proba(X_test)[:, 1]
    threshold = best_params["threshold"]
    test_pred = (test_probs >= threshold).astype(int)

    plot_path = os.path.join(RESULTS_DIR, "baseline_1_test_predictions.png")
    save_prediction_plot(
        y_true=y_test,
        y_probs=test_probs,
        threshold=threshold,
        path=plot_path
    )


    recall_test = recall_score(y_test, test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # --------------------------
    # Сохраняем результат
    # --------------------------
    recall_eval = best_params.pop("recall")
    result = {
        "eval_metrics": {
            "recall": recall_eval,
        },
        "best_params": best_params,
        "test_metrics": {
            "recall": recall_test,
            "fpr": fpr
        }
    }

    with open(os.path.join(RESULTS_DIR, "baseline_1.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("Done. Results saved to ./results/baseline_1.json")

main()
