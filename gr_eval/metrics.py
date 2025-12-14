from sklearn.metrics import recall_score, precision_score, roc_auc_score

def compute_metrics(y_true, y_pred, y_score=None):

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    metrics = {
        "recall": recall,
        "precision": precision,
        'f1': f1
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return metrics


def leakage_rate(y_true, y_pred):
    """Доля атак, которые прошли"""
    total_attacks = sum(y_true)
    leaked = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return leaked / total_attacks if total_attacks else 0
