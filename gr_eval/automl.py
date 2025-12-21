# gr_eval/automl.py
import optuna
import mlflow
from gr_eval.metrics import compute_metrics
from gr_eval.prompt_generator import PromptGenerator
from gr_eval.prompt_space import sample_prompt_spec

 
def automl_gr(llm, df_train, n_trials=30):

    prompt_generator = PromptGenerator(llm)
    
    def objective(trial):
        spec = sample_prompt_spec(trial)
        system_prompt = prompt_generator.generate(spec)

        y_true, y_score = [], []

        for _, row in df_train.iterrows():
            prompt = "\nUSER:\n" + row["text"]
            response = llm.generate(prompt, system_prompt=system_prompt)
            refused = llm.refused(response)

            y_true.append(1.0 if refused else 0.0)
            y_score.append(row["label"])

        metrics = compute_metrics(y_true, y_score)

        # 🔥 каждый trial — отдельный run
        with mlflow.start_run(nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("train_f1", metrics['f1'])
            mlflow.log_metric("train_precision", metrics['precision'])
            mlflow.log_metric("train_recall", metrics['recall'])
            mlflow.log_text(system_prompt, "system_prompt.txt")

        trial.set_user_attr("system_prompt", system_prompt)

        return metrics['f1']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study
