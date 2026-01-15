import os
import json
import numpy as np
import pandas as pd
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import ParameterGrid

from research.src.utils import load_data, find_best_threshold, save_prediction_plot, Embedder

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    # save
    with open(os.path.join(RESULTS_DIR, "baseline_1_model.pkl"),'wb') as f:
        pickle.dump(final_model, f)
    print("Done. Results saved to ./results/baseline_1.json")

main()
