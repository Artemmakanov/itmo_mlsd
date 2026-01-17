import os
import json
import itertools
import numpy as np
import pandas as pd
from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, roc_curve

import optuna

from research.src.evaluate import (
    metrics_on_segments,
    save_prediction_plot
)
from research.src.utils import load_data, Embedder


RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Feature helpers
# -------------------------
def compute_length_features(df: pd.DataFrame) -> np.ndarray:
    return df['text'].apply(len).values.reshape(-1, 1)


def compute_centroids(X: np.ndarray, y: np.ndarray):
    return {
        lbl: X[y == lbl].mean(axis=0)
        for lbl in np.unique(y)
    }


def build_features(
    df: pd.DataFrame,
    features: List[str],
    centroids=None,
    iso_model=None
) -> np.ndarray:

    arrs = []

    if "embedding" in features:
        X_emb = np.vstack(df["embedding"].values)
        arrs.append(X_emb)

    if "length_features" in features:
        arrs.append(compute_length_features(df))

    if "centroid_margin" in features:
        X = np.vstack(df["embedding"].values)
        pos_c = centroids[1]
        neg_c = centroids[0]

        margin = np.zeros((len(df), 1))
        for i, x in enumerate(X):
            dist_pos = np.linalg.norm(x - pos_c)
            dist_neg = np.linalg.norm(x - neg_c)
            margin[i, 0] = dist_neg - dist_pos  # FIX 3

        arrs.append(margin)

    if "anomaly_score" in features:
        scores = -iso_model.decision_function(
            np.vstack(df["embedding"].values)
        ).reshape(-1, 1)
        arrs.append(scores)

    return np.hstack(arrs)

# -------------------------
# Model factory
# -------------------------
def get_model(name, params):
    if name == "logreg":
        return LogisticRegression(random_state=42, **params)
    elif name == "LinearSVC":
        return LinearSVC(random_state=42, **params)
    elif name == "knn":
        return KNeighborsClassifier(**params)  # KNN не использует randomness
    elif name == "CatBoost":
        return CatBoostClassifier(random_state=42, verbose=0, **params)
    elif name == "IsolationForest":
        return IsolationForest(random_state=42, **params)
    else:
        raise ValueError(name)

# -------------------------
# Policy module
# -------------------------
def policy_module(score1, score2, t1, t2):
    decision = np.zeros_like(score1, dtype=int)
    for i in range(len(score1)):
        if score1[i] < t1:
            decision[i] = 0
        else:
            decision[i] = 1 if score2 is None or score2[i] >= t2 else 0
    return decision

# -------------------------
# Objective
# -------------------------
def objective_recall(trial, df_train, df_val, target_fpr=0.05):
    """
    Optuna objective: maximize Recall @ FPR <= 0.05 on eval set
    Stage2 optional, trained on Stage1 errors
    """
    feat_choices = ["embedding", "length_features", "anomaly_score", "centroid_margin"]

    # Все комбинации признаков
    all_combinations = []
    for n in range(1, len(feat_choices) + 1):
        for comb in itertools.combinations(feat_choices, n):
            all_combinations.append(",".join(comb))
    features = trial.suggest_categorical("features", all_combinations).split(",")

    use_scaler = trial.suggest_categorical("use_scaler", [True, False])

    stage1_name = trial.suggest_categorical(
        "stage1_model", ["logreg", "LinearSVC", "knn", "CatBoost"]
    )
    stage2_name = trial.suggest_categorical(
        "stage2_model", ["logreg", "LinearSVC", "knn", "CatBoost"]
    )

    stage1_params, stage2_params = {}, {}
    if stage1_name == "knn":
        stage1_params["n_neighbors"] = trial.suggest_int("knn_n", 1, 20)
    if stage1_name == "CatBoost":
        stage1_params.update(
            iterations=trial.suggest_int("cb_iter", 50, 200),
            depth=trial.suggest_int("cb_depth", 3, 6),
        )

    if stage2_name == "knn":
        stage2_params["n_neighbors"] = trial.suggest_int("knn2_n", 1, 20)
    if stage2_name == "CatBoost":
        stage2_params.update(
            iterations=trial.suggest_int("cb2_iter", 50, 200),
            depth=trial.suggest_int("cb2_depth", 3, 6),
        )

    trial.set_user_attr("stage1_name", stage1_name)
    trial.set_user_attr("stage1_params", stage1_params)
    trial.set_user_attr("stage2_name", stage2_name)
    trial.set_user_attr("stage2_params", stage2_params)
    trial.set_user_attr("features", features)
    trial.set_user_attr("use_scaler", use_scaler)

    y_train = df_train["label"].values
    y_val = df_val["label"].values

    # --- Optional features ---
    centroids, iso_model = None, None
    if "centroid_margin" in features:
        centroids = compute_centroids(np.vstack(df_train["embedding"].values), y_train)
    if "anomaly_score" in features:
        iso_model = IsolationForest(random_state=42).fit(np.vstack(df_train["embedding"].values))

    # --- Build features ---
    X_train = build_features(df_train, features, centroids, iso_model)
    X_val = build_features(df_val, features, centroids, iso_model)

    # --- Stage1 model ---
    model1 = get_model(stage1_name, stage1_params)
    if use_scaler:
        model1 = Pipeline([("scaler", StandardScaler()), ("model", model1)])
    model1.fit(X_train, y_train)

    if stage1_name == "LinearSVC":
        scores1_train = model1.decision_function(X_train)
        scores1_val = model1.decision_function(X_val)
    else:
        scores1_train = model1.predict_proba(X_train)[:, 1]
        scores1_val = model1.predict_proba(X_val)[:, 1]

    # -------------------------
    # Stage1 threshold via ROC (train)
    # -------------------------
    fpr1, tpr1, thresholds1 = roc_curve(y_train, scores1_train)
    # выберем наименьший threshold, где FPR <= 0.25
    target_fpr_stage1 = 0.25
    valid_idx = np.where(fpr1 <= target_fpr_stage1)[0]
    if len(valid_idx) == 0:
        best_thresh_stage1 = thresholds1[np.argmax(tpr1 - fpr1)]  # fallback
    else:
        best_thresh_stage1 = thresholds1[valid_idx[np.argmax(tpr1[valid_idx])]]

    pred_train_stage1 = (scores1_train >= best_thresh_stage1).astype(int)
    errors_idx = np.where(pred_train_stage1 != y_train)[0]

    if len(errors_idx) == 0 or len(np.unique(y_train[errors_idx])) < 2:
        # Stage2 не нужен
        trial.set_user_attr("stage2_is_needed", False)
        # final threshold grid search on eval
        try:
            _, _, _recall_eval = grid_search_thresholds(scores1_val, None, y_val, target_fpr=0.05, n_percentiles=25)
            return _recall_eval
        except:
            return 0.

    trial.set_user_attr("stage2_is_needed", True)

    # --- Stage2 model (train on Stage1 errors) ---
    model2 = get_model(stage2_name, stage2_params)
    if use_scaler:
        model2 = Pipeline([("scaler", StandardScaler()), ("model", model2)])
    model2.fit(X_train[errors_idx], y_train[errors_idx])

    if stage2_name == "LinearSVC":
        scores2_val = model2.decision_function(X_val)
    else:
        scores2_val = model2.predict_proba(X_val)[:, 1]

    # --- final threshold search for Stage1+Stage2 policy ---
    try:
        t1, t2, best_recall_eval = grid_search_thresholds(scores1_val, scores2_val, y_val, target_fpr=0.05, n_percentiles=25)
        return best_recall_eval
    except:
        return 0.


def grid_search_thresholds(scores1, scores2, y_true, target_fpr=0.05, n_percentiles=25):
    best_recall, best_t1, best_t2 = 0, None, None
    # Compute thresholds as percentiles
    thresholds1 = np.percentile(scores1, np.linspace(0, 100, n_percentiles))
    thresholds2 = np.percentile(scores2, np.linspace(0, 100, n_percentiles)) if scores2 is not None else [None]

    for t1 in thresholds1:
        for t2 in thresholds2:
            y_pred = policy_module(scores1, scores2, t1, t2)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr_val = fp / (fp + tn)
            if fpr_val <= target_fpr:
                recall = tp / (tp + fn)
                if recall > best_recall:
                    best_recall = recall
                    best_t1, best_t2 = t1, t2

    if best_t1 is None:
        raise Exception(f"FPR is always > {target_fpr}")

    return best_t1, best_t2, best_recall


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    train_df = load_data("train")
    eval_df = load_data("eval")
    test_df = load_data("test")

    embedder = Embedder()
    for df in [train_df, eval_df, test_df]:
        df["embedding"] = list(embedder.encode(df["text"].tolist()))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: objective_recall(t, train_df, eval_df), n_trials=50)

    best = study.best_trial

    stage1_name = best.user_attrs["stage1_name"]
    stage1_params = best.user_attrs["stage1_params"]
    stage2_name = best.user_attrs["stage2_name"]
    stage2_params = best.user_attrs["stage2_params"]
    features = best.user_attrs["features"]
    use_scaler = best.user_attrs["use_scaler"]
    stage2_is_needed = best.user_attrs["stage2_is_needed"]

    y_train = train_df["label"].values
    y_eval = eval_df["label"].values
    y_test = test_df["label"].values

    centroids = None
    iso_model = None

    if "centroid_margin" in features:
        centroids = compute_centroids(
            np.vstack(train_df["embedding"].values), y_train
        )

    if "anomaly_score" in features:
        iso_model = IsolationForest(random_state=42).fit(
            np.vstack(train_df["embedding"].values)
        )

    X_train = build_features(train_df, features, centroids, iso_model)
    X_eval = build_features(eval_df, features, centroids, iso_model)
    X_test = build_features(test_df, features, centroids, iso_model)

    model1 = get_model(stage1_name, stage1_params)
    if use_scaler:
        model1 = Pipeline([("scaler", StandardScaler()), ("model", model1)])
    model1.fit(X_train, y_train)

    if stage1_name == "LinearSVC":
        scores1_train = model1.decision_function(X_train)
        scores1_eval = model1.decision_function(X_eval)
        scores1_test = model1.decision_function(X_test)
    else:
        scores1_train = model1.predict_proba(X_train)[:, 1]
        scores1_eval = model1.predict_proba(X_eval)[:, 1]
        scores1_test = model1.predict_proba(X_test)[:, 1]

    best_recall_eval = None  # FIX 4

    if stage2_is_needed:

        fpr, tpr, thresholds = roc_curve(y_train, scores1_train)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        errors_idx = np.where((scores1_train >= best_thresh).astype(int) != y_train)[0]

        model2 = get_model(stage2_name, stage2_params)
        if use_scaler:
            model2 = Pipeline([("scaler", StandardScaler()), ("model", model2)])
        model2.fit(X_train[errors_idx], y_train[errors_idx])  # FIX 1

        if stage2_name == "LinearSVC":
            scores2_eval = model2.decision_function(X_eval)
            scores2_test = model2.decision_function(X_test)
        else:
            scores2_eval = model2.predict_proba(X_eval)[:, 1]
            scores2_test = model2.predict_proba(X_test)[:, 1]

        t1, t2, best_recall_eval = grid_search_thresholds(scores1_eval, scores2_eval, y_eval, target_fpr=0.05, n_percentiles=25)

        save_prediction_plot(
            y_true=y_test,
            y_probs=scores1_test,
            threshold=t1,
            path=os.path.join(RESULTS_DIR, "mvp_stage1_predictions.png")
        )

        save_prediction_plot(
            y_true=y_test,
            y_probs=scores2_test,
            threshold=t2,
            path=os.path.join(RESULTS_DIR, "mvp_stage2_predictions.png")
        )

        test_pred = policy_module(scores1_test, scores2_test, t1, t2)

    else:
        t1, _, best_recall_eval = grid_search_thresholds(scores1_eval, None, y_eval, target_fpr=0.05, n_percentiles=25)

        save_prediction_plot(
            y_true=y_test,
            y_probs=scores1_test,
            threshold=t1,
            path=os.path.join(RESULTS_DIR, "mvp_stage1_predictions.png")
        )

        test_pred = (scores1_test >= t1).astype(int)

    test_overall_metrics = metrics_on_segments(
        df=test_df,
        y_true=y_test,
        y_pred=test_pred,
        segment_names=["Blind Spot", "Naturally Allowed", "Obviously Refused"]
    )

    test_blind_spots_metrics = metrics_on_segments(
        df=test_df,
        y_true=y_test,
        y_pred=test_pred,
        segment_names=["Blind Spot", "Naturally Allowed"]
    )

    with open(os.path.join(RESULTS_DIR, "mvp.json"), "w") as f:
        json.dump({
            "best_params": best.params,
            "stage2_is_needed": stage2_is_needed,
            "eval_overall_recall": best_recall_eval,
            "test_metrics": {
                "overall": test_overall_metrics,
                "blind_spots": test_blind_spots_metrics
            }
        }, f, indent=2)

    print("Done.")
