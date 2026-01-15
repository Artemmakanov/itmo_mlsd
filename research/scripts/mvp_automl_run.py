import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score
import optuna
import pickle
from research.src.utils import load_data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, confusion_matrix
from research.src.utils import load_data, find_best_threshold, save_prediction_plot, Embedder

# -----------------------------
# CONFIG
# -----------------------------
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# -----------------------------
# UTILITIES
# -----------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def join_features(rows, labeling):
    for i, row in enumerate(rows):
        try:
            row_labeling = labeling[i]
            features = {field: value for field, value in row_labeling.items() if field not in ['text', 'id']}
            row.update(features)
        except:
            None
    return rows

def preprocess_features(records, embedder):
    """
    Превращает JSON-фичи в числовую матрицу для AutoML.
    Булевы -> int, categorical -> one-hot, плюс embeddings текста.
    """
    df = pd.DataFrame(records)
    df['dangerous_variant'] = df['dangerous_variant'].fillna('')
    X_emb_dangerous = embedder.encode(df['dangerous_variant'].tolist())  # shape (N, embedding_dim)
    X_emb_text = embedder.encode(df['text'].tolist())  # shape (N, embedding_dim)
    X_emb_dot = np.sum(X_emb_text * X_emb_dangerous, axis=1, keepdims=True)


    df = df.fillna(0)

    # ---- Бинарные фичи ----
    bool_cols = [
        "explicit_harm",
        "operational_detail",
        "dual_use",
        "benign_context",
        "can_non_expert_execute",
        "contrast_less_harmful",
    ]
    for col in bool_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].astype(int)

    # ---- Instructiveness score ----
    if "instructiveness_score" not in df.columns:
        df["instructiveness_score"] = 0

    # ---- required_tools_level -> one-hot ----
    df["required_tools_level"] = df.get("required_tools_level", "")
    for lvl in ["none", "common", "specialized"]:
        df[f"tools_{lvl}"] = (df["required_tools_level"] == lvl).astype(int)

    # ---- intent -> one-hot ----
    df["intent"] = df.get("intent", "")
    for intent in ["academic", "curiosity", "fictional", "practical_real_world"]:
        df[f"intent_{intent}"] = (df["intent"] == intent).astype(int)

    # ---- числовая матрица фич ----
    feature_cols = bool_cols + ["instructiveness_score"] + \
                   [c for c in df.columns if c.startswith("tools_") or c.startswith("intent_")]

    X_numeric = df[feature_cols].values

    # ---- эмбеддинги ----
    # load
    # with open(os.path.join(RESULTS_DIR, "baseline_1_model.pkl"), 'rb') as f:
    #     knn = pickle.load(f)

    # knn_signal = knn.predict_proba(X_emb)[:, 1].reshape(-1, 1)  
    # ---- объединяем все фичи ----
    # X = np.hstack([X_numeric, X_emb, knn_signal])
    # X = np.hstack([X_numeric, X_emb_dangerous, X_emb_text])
    X = np.hstack([X_numeric, X_emb_dot, X_emb_text])
    return X, df


train_labeling = load_jsonl( "data/train_llm_markup.jsonl")
eval_labeling = load_jsonl( "data/eval_llm_markup.jsonl")
test_labeling = load_jsonl("data/test_llm_markup.jsonl")

train = load_data('train').to_dict(orient="records")
eval = load_data('eval').to_dict(orient="records")
test = load_data('test').to_dict(orient="records")

train = join_features(train, train_labeling)
eval = join_features(eval, eval_labeling)
test = join_features(test, test_labeling)

embedder = Embedder()

# -----------------------------
# Features & labels
# -----------------------------
X_train, df_train = preprocess_features(train, embedder)
y_train = df_train["label"].values

X_eval, df_eval = preprocess_features(eval, embedder)
y_eval = df_eval["label"].values

X_test, df_test = preprocess_features(test, embedder)
y_test = df_test["label"].values

# Hyperparameter search with threshold optimization
# -----------------------------

# Берём, например, 80% train, 20% eval
# from sklearn.model_selection import train_test_split

# X_train, X_eval, y_train, y_eval = train_test_split(
#     X_eval, y_eval, test_size=0.5, random_state=42, stratify=y_eval
# )

def objective(trial):
    hidden_layer_sizes = tuple([trial.suggest_int(f"layer_{i}", 16, 128) 
                                for i in range(trial.suggest_int("n_layers", 1, 3))])
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
    alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=50,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Подбираем threshold на eval
    y_probs = model.predict_proba(X_eval)[:, 1]
    # try:
    #     threshold = find_best_threshold(y_eval, y_probs)
    # except Exception:
    #     return 0.

    # y_pred = (y_probs >= threshold).astype(int)
    # recall = recall_score(y_eval, y_pred)

    # tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
    # fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # # print(recall, fpr)
    # return recall - fpr # Optuna maximize
    return roc_auc_score(y_eval, y_probs, )

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best params:", best_params)

# -----------------------------
# Train final MLP model on train + eval
# -----------------------------
n_layers = best_params["n_layers"]
hidden_layer_sizes = tuple([best_params[f"layer_{i}"] for i in range(n_layers)])

final_model = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=best_params["activation"],
    alpha=best_params["alpha"],
    learning_rate_init=best_params["learning_rate_init"],
    max_iter=50,
    random_state=42
)

X_final = np.vstack([X_train, X_eval])
y_final = np.hstack([y_train, y_eval])
final_model.fit(X_final, y_final)

# -----------------------------
# Test metrics с threshold
# -----------------------------
y_probs_final = final_model.predict_proba(X_final)[:, 1]
threshold_final = find_best_threshold(y_final, y_probs_final)

y_probs_test = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_probs_test >= threshold_final).astype(int)

recall_test = recall_score(y_test, y_test_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
fpr = fp / (fp + tn)

print("Test recall:", recall_test, "FPR:", fpr)

# -----------------------------
# Save results
# -----------------------------
result = {
    "eval_metrics": {"recall": study.best_value},
    "best_params": best_params,
    "test_metrics": {"recall": recall_test, "fpr": fpr},
    "threshold": threshold_final
}

import os, json
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "mvp.json"), "w") as f:
    json.dump(result, f, indent=2)

plot_path = os.path.join(RESULTS_DIR, "mvp_test_predictions.png")
save_prediction_plot(
    y_true=y_test,
    y_probs=y_probs_test,
    threshold=threshold_final,
    path=plot_path
)

print("Done. Results saved to mvp.json")