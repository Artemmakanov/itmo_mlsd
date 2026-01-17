import os
import json
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import ParameterGrid

from research.src.utils import load_data, Embedder
from research.src.evaluate import metrics_on_segments, find_best_threshold, save_prediction_plot

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------
# Grid search KNN
# --------------------------
def grid_search_knn(X_train, y_train, X_eval, y_eval, param_grid, log_every=20):
    grid = list(ParameterGrid(param_grid))
    total = len(grid)

    print(f"Starting KNN grid search | total configs = {total}")

    best_score = -1.0
    best_params = None

    for i, params in enumerate(grid, 1):
        model = KNeighborsClassifier(**params)
        model.fit(X_train, y_train)

        y_probs = model.predict_proba(X_eval)[:, 1]

        try:
            threshold = find_best_threshold(y_eval, y_probs)
        except Exception as e:
            print(f"Threshold search failed | params={params} | error={e}")
            continue

        y_pred = (y_probs >= threshold).astype(int)
        overall_recall = recall_score(y_eval, y_pred)

        # периодический прогресс
        if i % log_every == 0 or i == 1:
            print(
                f"[{i}/{total}] recall={overall_recall:.4f} | "
                f"n_neighbors={params['n_neighbors']} | "
                f"weights={params['weights']} | "
                f"metric={params['metric']}"
            )

        # новый лучший результат
        if overall_recall > best_score:
            best_score = overall_recall
            best_params = {
                **params,
                "threshold": threshold,
                "overall_recall": best_score
            }

            print(
                f"NEW BEST ✔ recall={best_score:.4f} | "
                f"params={params} | threshold={threshold:.4f}"
            )

    print(
        f"Grid search finished | best recall={best_score:.4f} | best params={best_params}"
    )

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
        "n_neighbors": (
            list(range(3, 15, 2)) +      # локальная структура
            list(range(20, 101, 20)) +   # средний масштаб
            [150, 200, 300]              # глобальный контекст
        ),
        "weights": ["distance", "uniform"],
        "metric": ["cosine"]
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

    eval_overall_recall = best_params["overall_recall"]

    test_overall_metrics = metrics_on_segments(
        df=test_df,
        y_true=y_test,
        y_pred=test_pred,
        segment_names=[
            "Blind Spot",
            "Naturally Allowed",
            "Obviously Refused"
        ]
    )

    test_blind_spots_metrics = metrics_on_segments(
        df=test_df,
        y_true=y_test,
        y_pred=test_pred,
        segment_names=[
            "Blind Spot",
            "Naturally Allowed",
        ]
    )

    # --------------------------
    # Сохраняем результат
    # --------------------------

    result = {
        "automl_metric": "recall_overall",
        "eval_overall_recall": eval_overall_recall,
        "best_params": best_params,
        "test_metrics": {
            "overall": test_overall_metrics,
            "blind_spots": test_blind_spots_metrics
        }
    }


    with open(os.path.join(RESULTS_DIR, "baseline_1.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("Done. Results saved to ./results/baseline_1.json")

main()
