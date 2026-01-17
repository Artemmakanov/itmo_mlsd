import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, recall_score


def metrics_on_segments(df, y_true, y_pred, segment_names):
    seg_mask = df["segment"].isin(segment_names)

    if seg_mask.sum() == 0:
        return {
            "recall": 0.0,
            "fpr": 0.0,
            "support": 0
        }

    y_t = y_true[seg_mask.values]
    y_p = y_pred[seg_mask.values]

    recall = recall_score(y_t, y_p)

    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "recall": recall,
        "fpr": fpr,
        "support": int(seg_mask.sum())
    }

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
        label="y=0 (Allowed)"
    )
    plt.hist(
        pos, bins=50, alpha=0.6, density=True,
        label="y=1 (Refusal)"
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
