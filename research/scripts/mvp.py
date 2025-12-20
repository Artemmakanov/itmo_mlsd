import sys
sys.path.append("../..")

import pandas as pd
import mlflow

from gr_eval.automl import automl_gr
from gr_eval.metrics import compute_metrics
from gr_eval.llm import HFSmallLLM

train = pd.read_csv(r'../../data/train.csv',)
test = pd.read_csv(r'../../data/test.csv', )
model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
llm = HFSmallLLM(device='cuda', model_name=model_name)


mlflow.set_experiment("gr_llm_eval")

with mlflow.start_run(run_name="automl_gr_tpe"):
    study = automl_gr(
        llm=llm,
        df_train=train,
        n_trials=1,
    )

    best_prompt = study.best_trial.user_attrs["system_prompt"]
    best_score = study.best_value

    mlflow.log_metric("best_train_f1", best_score)
    mlflow.log_text(best_prompt, "best_system_prompt.txt")

    print("Best F1 (train):", best_score)
    print("Best system prompt:\n", best_prompt)


y_true, y_pred, y_score = [], [], []

for _, row in test.iterrows():
    full_prompt = best_prompt + "\nUSER:\n" + row["text"]
    response = llm.generate(full_prompt)
    refused = llm.refused(response)

    y_pred.append(1 if refused else 0)
    y_score.append(1.0 if refused else 0.0)
    y_true.append(row["label"])

final_metrics = compute_metrics(y_true, y_pred, y_score)
print("LLM + AutoML GR (test):", final_metrics)
