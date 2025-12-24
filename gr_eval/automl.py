# gr_eval/automl.py
import optuna
import mlflow
from gr_eval.metrics import compute_metrics
from gr_eval.prompt_generator import PromptGenerator
from gr_eval.fewshot_selector import FewShotSelector
from gr_eval.error_store import ErrorStore
from gr_eval.prompt_space import sample_prompt_spec

 
def automl_gr(llm, df_train, n_trials=30):

    error_store = ErrorStore(max_iterations=n_trials)
    fewshot_selector = FewShotSelector()
    prompt_generator = PromptGenerator(llm)
    
    def objective(trial):
        fn_hist, fp_hist = error_store.get_prev_iteration(trial.number)

        few_shots = fewshot_selector.select(
            fn_hist, fp_hist,
            num_fn=3, num_fp=3,
        )

        spec = sample_prompt_spec(trial)
        system_prompt = prompt_generator.generate(spec, few_shots=few_shots)

        y_true, y_score = [], []

        for _, row in df_train.iterrows():
            label = row["label"]
            prompt = row["text"]

            response = llm.generate(prompt, system_prompt=system_prompt)
            refused = llm.refused(response)
            pred = 1.0 if refused else 0.0

            y_true.append(pred)
            y_score.append(label)
            error_store.add(trial.number, prompt, label, pred)
            print("="*80)
            print(f"prompt = {prompt}")
            print(f"response = {response}")
            print(f"label = {label}")
            print(f"pred = {refused}")

        metrics = compute_metrics(y_true, y_score)
        metrics["iteration"] = trial.number
        metrics["num_fn_seen"] = len(fn_hist)
        metrics["num_fp_seen"] = len(fp_hist)

        # 🔥 каждый trial — отдельный run
        with mlflow.start_run(nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_text(system_prompt, "system_prompt.txt")
            for name, value in metrics.items():
                mlflow.log_metric(f"train_{name}", value)

        trial.set_user_attr("system_prompt", system_prompt)

        return metrics['f1']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study
