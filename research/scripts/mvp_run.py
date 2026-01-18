import os
import json
import itertools
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import optuna

from research.src.evaluate import metrics_on_segments, save_prediction_plot
from research.src.utils import load_data, Embedder
from research.src.data import build_features, compute_centroids

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Model factory
# -------------------------
def get_model(name, params):
    if name == "logreg":
        return LogisticRegression(random_state=42, **params)
    elif name == "LinearSVC":
        return LinearSVC(random_state=42, **params)
    elif name == "knn":
        return KNeighborsClassifier(**params)
    elif name == "CatBoost":
        return CatBoostClassifier(random_state=42, verbose=0, **params)
    elif name == "IsolationForest":
        return IsolationForest(random_state=42, **params)
    elif name == "SVC":
        return SVC(random_state=42, **params)
    elif name == "mlp":
        return MLPClassifier(random_state=42, early_stopping=True, **params)
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
# Threshold search (percentiles)
# -------------------------
def grid_search_thresholds(scores1, scores2, y_true, target_metric=0.05, n_percentiles=25):
    best_metric, best_t1, best_t2 = 0, None, None
    thresholds1 = np.percentile(scores1, np.linspace(0, 100, n_percentiles))
    thresholds2 = np.percentile(scores2, np.linspace(0, 100, n_percentiles)) if scores2 is not None else [None]

    for t1 in thresholds1:
        for t2 in thresholds2:
            y_pred = policy_module(scores1, scores2, t1, t2)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr_val = fp / (fp + tn)
            recall_val = tp / (tp + fn)
            if scores2 is None:
                metric = recall_val  # Stage1 objective: Recall@FPR
                if fpr_val <= target_metric and metric > best_metric:
                    best_metric = metric
                    best_t1, best_t2 = t1, t2
            else:
                metric = 1 - fpr_val  # Stage2 objective: minimize FPR @ Recall >= R_low
                if recall_val >= target_metric and metric > best_metric:
                    best_metric = metric
                    best_t1, best_t2 = t1, t2

    if best_t1 is None:
        raise Exception(f"No thresholds satisfy criteria with target_metric={target_metric}")

    return best_t1, best_t2, best_metric


def objective_stage1(trial, df_train, df_val, target_fpr):
    feat_choices = ["embedding", "length_features", "anomaly_score", "centroid_margin", "pos", "keywords"]

    # --- Feature selection ---
    all_combinations = []
    for n in range(1, len(feat_choices) + 1):
        for comb in itertools.combinations(feat_choices, n):
            all_combinations.append(",".join(comb))
    features = trial.suggest_categorical("features", all_combinations).split(",")

    use_scaler = trial.suggest_categorical("use_scaler", [True, False])

    stage1_name = trial.suggest_categorical(
        "stage1_model", ["logreg", "LinearSVC", "knn", "CatBoost", "SVC", "mlp"]
    )

    stage1_params = {}
    if stage1_name == "knn":
        stage1_params["n_neighbors"] = trial.suggest_int("knn_n", 1, 20)
    if stage1_name == "CatBoost":
        stage1_params.update(
            iterations=trial.suggest_int("cb_iter", 50, 200),
            depth=trial.suggest_int("cb_depth", 3, 6),
        )
    if stage1_name == "mlp":
        n_layers = trial.suggest_int("mlp_n_layers", 1, 2)

        if n_layers == 1:
            hidden = (
                trial.suggest_int("mlp_hidden_1", 50, 200),
            )
        else:
            hidden = (
                trial.suggest_int("mlp_hidden_1", 50, 200),
                trial.suggest_int("mlp_hidden_2", 20, 150),
            )

        stage1_params.update(
            hidden_layer_sizes=hidden,
            max_iter=trial.suggest_int("mlp_max_iter", 50, 150),
        )

    trial.set_user_attr("stage1_name", stage1_name)
    trial.set_user_attr("stage1_params", stage1_params)
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

    if stage1_name in ["LinearSVC", "SVC"]:
        scores1_val = model1.decision_function(X_val)
    else:
        scores1_val = model1.predict_proba(X_val)[:, 1]

    # Threshold search Stage1 @ FPR <= 0.3
    try:
        t1, _, recall_stage1 = grid_search_thresholds(scores1_val, None, y_val, target_metric=target_fpr, n_percentiles=25)
        trial.set_user_attr("stage1_threshold", t1)
        trial.set_user_attr("stage1_recall", recall_stage1)
        return recall_stage1
    except:
        trial.set_user_attr("stage1_threshold", None)
        trial.set_user_attr("stage1_recall", 0.)
        return 0.

# -------------------------
def objective_stage2(trial, df_train, df_val, X_train, X_val, stage1_best_trial, target_fpr, recall_degradation,
                     model1=None, scores1_train=None, scores1_val=None):
    """
    model1, scores1_train, scores1_val — передаем уже обученную Stage1 модель и её предсказания,
    чтобы не учить Stage1 заново.
    """
    features = stage1_best_trial.user_attrs["features"]
    use_scaler = stage1_best_trial.user_attrs["use_scaler"]
    stage1_name = stage1_best_trial.user_attrs["stage1_name"]
    stage1_params = stage1_best_trial.user_attrs["stage1_params"]
    R = stage1_best_trial.user_attrs["stage1_recall"]
    R_low = R * recall_degradation

    stage2_name = trial.suggest_categorical(
        "stage2_model", ["logreg", "LinearSVC", "knn", "CatBoost", "SVC", "mlp"]
    )
    stage2_params = {}
    if stage2_name == "knn":
        stage2_params["n_neighbors"] = trial.suggest_int("knn2_n", 1, 20)
    if stage2_name == "CatBoost":
        stage2_params.update(
            iterations=trial.suggest_int("cb2_iter", 50, 200),
            depth=trial.suggest_int("cb2_depth", 3, 6),
        )
    if stage2_name == "mlp":
        n_layers = trial.suggest_int("mlp_n_layers", 1, 2)
        if n_layers == 1:
            hidden = (
                trial.suggest_int("mlp_hidden_1", 50, 200),
            )
        else:
            hidden = (
                trial.suggest_int("mlp_hidden_1", 50, 200),
                trial.suggest_int("mlp_hidden_2", 20, 150),
            )
        stage2_params.update(
            hidden_layer_sizes=hidden,
            max_iter=trial.suggest_int("mlp_max_iter", 50, 150),
        )

    trial.set_user_attr("stage2_name", stage2_name)
    trial.set_user_attr("stage2_params", stage2_params)

    y_train = df_train["label"].values
    y_val = df_val["label"].values

    # Threshold Stage1
    t1, _, _ = grid_search_thresholds(scores1_val, None, y_val, target_metric=target_fpr, n_percentiles=25)

    # Ошибки Stage1
    pred_train_stage1 = (scores1_train >= t1).astype(int)
    indices = np.where(pred_train_stage1 == True)[0]

    if len(indices) == 0 or len(np.unique(y_train[indices])) < 2:
        trial.set_user_attr("stage2_is_needed", False)
        return 0.  # Stage2 не нужен

    trial.set_user_attr("stage2_is_needed", True)

    # --- Stage2 model ---
    model2 = get_model(stage2_name, stage2_params)
    if use_scaler:
        model2 = Pipeline([("scaler", StandardScaler()), ("model", model2)])
    model2.fit(X_train[indices], y_train[indices])

    if stage2_name in ["LinearSVC", "SVC"]:
        scores2_val = model2.decision_function(X_val)
    else:
        scores2_val = model2.predict_proba(X_val)[:, 1]

    # Threshold search Stage1+Stage2 policy @ Recall >= R_low
    t1_final, t2_final, fpr_metric = grid_search_thresholds(scores1_val, scores2_val, y_val, target_metric=R_low, n_percentiles=25)

    trial.set_user_attr("stage2_thresholds", (t1_final, t2_final))
    trial.set_user_attr("stage2_fpr_metric", fpr_metric)

    return fpr_metric


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    target_fpr = 0.15
    recall_degradation = 0.95
    n_trials1 = 50
    n_trials2 = 50
    n_percentiles_final = 50
    # --- Load data & embeddings ---
    train_df = load_data("train")
    eval_df = load_data("eval")
    test_df = load_data("test")

    embedder = Embedder()
    for df in [train_df, eval_df, test_df]:
        df["embedding"] = list(embedder.encode(df["text"].tolist()))

    # -------------------------
    # Stage1 optimization
    # -------------------------
    study_stage1 = optuna.create_study(direction="maximize",
                                       sampler=optuna.samplers.TPESampler(seed=42))
    study_stage1.optimize(lambda t: objective_stage1(t, train_df, eval_df, target_fpr=target_fpr), n_trials=n_trials1)
    best_stage1 = study_stage1.best_trial

    stage1_name = best_stage1.user_attrs["stage1_name"]
    stage1_params = best_stage1.user_attrs["stage1_params"]
    features = best_stage1.user_attrs["features"]
    use_scaler = best_stage1.user_attrs["use_scaler"]
    stage1_threshold = best_stage1.user_attrs["stage1_threshold"]
    stage1_recall = best_stage1.user_attrs["stage1_recall"]
    R_low = stage1_recall * recall_degradation

    print(f"Stage1 best: model={stage1_name}, features={features}, threshold={stage1_threshold:.4f}, recall={stage1_recall:.4f}")


    y_train = train_df["label"].values
    y_eval = eval_df["label"].values
    y_test = test_df["label"].values

    centroids = compute_centroids(np.vstack(train_df["embedding"].values), y_train) if "centroid_margin" in features else None
    iso_model = IsolationForest(random_state=42).fit(np.vstack(train_df["embedding"].values)) if "anomaly_score" in features else None


    X_train = build_features(train_df, features, centroids, iso_model)
    X_eval = build_features(eval_df, features, centroids, iso_model)
    X_test = build_features(test_df, features, centroids, iso_model)

    
    # Получаем Stage1 model и scores один раз
    model1_stage1 = get_model(stage1_name, stage1_params)
    if use_scaler:
        model1_stage1 = Pipeline([("scaler", StandardScaler()), ("model", model1_stage1)])
    model1_stage1.fit(X_train, y_train)
    if stage1_name in ["LinearSVC", "SVC"]:
        scores1_train = model1_stage1.decision_function(X_train)
        scores1_val = model1_stage1.decision_function(X_eval)
    else:
        scores1_train = model1_stage1.predict_proba(X_train)[:, 1]
        scores1_val = model1_stage1.predict_proba(X_eval)[:, 1]

    study_stage2 = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_stage2.optimize(
        lambda t: objective_stage2(
            t, train_df, eval_df, X_train, X_eval, best_stage1, target_fpr=target_fpr,
            recall_degradation=recall_degradation,
            model1=model1_stage1,
            scores1_train=scores1_train,
            scores1_val=scores1_val
        ),
        n_trials=n_trials2
    )
    best_stage2 = study_stage2.best_trial

    stage2_needed = best_stage2.user_attrs.get("stage2_is_needed", False)

    if stage2_needed:
        stage2_name = best_stage2.user_attrs["stage2_name"]
        stage2_params = best_stage2.user_attrs["stage2_params"]
        t1_final, t2_final = best_stage2.user_attrs["stage2_thresholds"]
        stage2_fpr_metric = best_stage2.user_attrs["stage2_fpr_metric"]
        print(f"Stage2 best: model={stage2_name}, thresholds=({t1_final:.4f},{t2_final:.4f}), fpr_metric={stage2_fpr_metric:.4f}")
    else:
        stage2_name = None
        stage2_params = None
        t1_final = stage1_threshold
        t2_final = None
        stage2_fpr_metric = None
        print("Stage2 not needed.")

    # --- Stage1 model ---
    model1 = get_model(stage1_name, stage1_params)
    if use_scaler:
        model1 = Pipeline([("scaler", StandardScaler()), ("model", model1)])
    model1.fit(X_train, y_train)

    if stage1_name in ["LinearSVC", "SVC"]:
        scores1_train = model1.decision_function(X_train)
        scores1_eval = model1.decision_function(X_eval)
        scores1_test = model1.decision_function(X_test)
    else:
        scores1_train = model1.predict_proba(X_train)[:, 1]
        scores1_eval = model1.predict_proba(X_eval)[:, 1]
        scores1_test = model1.predict_proba(X_test)[:, 1]

    pred_test_stage1 = (scores1_test >= stage1_threshold).astype(int)

    # --- Stage2 model (if needed) ---
    if stage2_needed:
        pred_train_stage1 = (scores1_train >= stage1_threshold).astype(int)
        # errors_idx = np.where((pred_train_stage1 != y_train) & ((y_train == 1) | (y_train == 0)))[0]
        indices = np.where((pred_train_stage1 == True))[0]

        model2 = get_model(stage2_name, stage2_params)
        if use_scaler:
            model2 = Pipeline([("scaler", StandardScaler()), ("model", model2)])
        model2.fit(X_train[indices], y_train[indices])

        if stage2_name in ["LinearSVC", "SVC"]:
            scores2_eval = model2.decision_function(X_eval)
            scores2_test = model2.decision_function(X_test)
        else:
            scores2_eval = model2.predict_proba(X_eval)[:, 1]
            scores2_test = model2.predict_proba(X_test)[:, 1]

        # --- Threshold search final ---
        t1_final, t2_final, _ = grid_search_thresholds(scores1_eval, scores2_eval, y_eval, target_metric=R_low, n_percentiles=n_percentiles_final)

        save_prediction_plot(
            y_true=y_test,
            y_probs=scores1_test,
            threshold=t1_final,
            path=os.path.join(RESULTS_DIR, "mvp_stage1_predictions.png")
        )
        save_prediction_plot(
            y_true=y_test,
            y_probs=scores2_test,
            threshold=t2_final,
            path=os.path.join(RESULTS_DIR, "mvp_stage2_predictions.png")
        )

        test_pred = policy_module(scores1_test, scores2_test, t1_final, t2_final)

    else:
        t1_final = stage1_threshold
        save_prediction_plot(
            y_true=y_test,
            y_probs=scores1_test,
            threshold=t1_final,
            path=os.path.join(RESULTS_DIR, "stage1_predictions.png")
        )
        test_pred = (scores1_test >= t1_final).astype(int)

    # -------------------------
    # Evaluate final predictions
    # -------------------------
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

    test_overall_stage1_only_metrics = metrics_on_segments(
        df=test_df,
        y_true=y_test,
        y_pred=pred_test_stage1,
        segment_names=["Blind Spot", "Naturally Allowed", "Obviously Refused"]
    )

    # -------------------------
    # Save results
    # -------------------------
    with open(os.path.join(RESULTS_DIR, "mvp.json"), "w") as f:
        json.dump({
            "stage1_best": {
                "model": stage1_name,
                "params": stage1_params,
                "features": features,
                "threshold": t1_final,
                "recall_eval": stage1_recall
            },
            "stage2": {
                "needed": stage2_needed,
                "model": stage2_name,
                "params": stage2_params,
                "thresholds": (t1_final, t2_final),
                "fpr_metric": stage2_fpr_metric
            },
            "test_metrics": {
                "overall": test_overall_metrics,
                "blind_spots": test_blind_spots_metrics,
                "stage1_only": test_overall_stage1_only_metrics
            }
        }, f, indent=2)

    print("✅ Pipeline completed.")
