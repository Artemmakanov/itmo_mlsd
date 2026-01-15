import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from sklearn.metrics import precision_recall_curve, confusion_matrix

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
# Эмбеддинги через MiniLM
# --------------------------
class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.encoder.eval()

    def encode(self, texts):
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return embeddings
