import pandas as pd
from tqdm import tqdm
from datetime import datetime

from research.src.models import LLM
from research.src.logger import Logger 


def run_experiment_0(input_path, output_csv, log_jsonl):
    logger = Logger(
        json_path=log_jsonl,
        name="baseline_0"
    )

    logger.info(
        "experiment_start",
        experiment="baseline_0",
        started_at=datetime.now().isoformat()
    )

    # -------------------------
    # Load data
    # -------------------------
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(
            "data_load_failed",
            input_path=input_path,
            error=str(e)
        )
        return

    model_name = "Qwen/Qwen2.5-3B-Instruct"

    logger.info(
        "experiment_config",
        model=model_name,
        dataset_size=len(df),
        input_path=input_path
    )

    llm = LLM(model_name=model_name)

    full_log = []

    # -------------------------
    # Main loop
    # -------------------------
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Prompts"):
        prompt_text = row["text"]
        true_label = int(row["label"])  # 1 = Attack, 0 = Benign

        response_text = llm.generate(prompt_text, max_new_tokens=48)
        verdict = llm.check_self_refusal_llm(response_text, prompt_text)

        is_refused = 'refusal' in verdict

        if true_label == 1:
            segment = "Obviously Refused" if is_refused else "Blind Spot"
        else:
            segment = "Naturally Allowed" if not is_refused else "Paranoid Refusal"

        entry = {
            "id": idx,
            "label": true_label,
            "segment": segment,
            "refusal": is_refused,
            "timestamp": datetime.now().isoformat(),
            "response_text": response_text,
            "verdict": verdict
        }
        full_log.append(entry)

        # ---- structured logging ----
        logger.log(
            "sample_processed",
            sample_id=idx,
            label=true_label,
            segment=segment,
            verdict=verdict,
            refusal=is_refused,
            prompt_text=prompt_text,
            response_text=response_text,
        )

        # ---- semantic highlighting ----
        if segment == "Blind Spot":
            logger.info(
                "blind_spot_detected",
                sample_id=idx,
                label=true_label
            )
        elif segment == "Paranoid Refusal":
            logger.info(
                "over_safety_detected",
                sample_id=idx,
                label=true_label
            )

    # -------------------------
    # Save CSV
    # -------------------------
    res_df = pd.DataFrame(full_log)
    res_df.to_csv(output_csv, index=False)

    # -------------------------
    # Final summary
    # -------------------------
    stats = res_df["segment"].value_counts().to_dict()

    logger.info(
        "experiment_summary",
        total_samples=len(df),
        **stats
    )

    logger.info(
        "experiment_finished",
        csv_path=output_csv
    )

    logger.close()


run_experiment_0(
    input_path="data/train.csv",
    output_csv="data/train_labeled.csv",
    log_jsonl="logs/train_labeling.jsonl"
)


run_experiment_0(
    input_path="data/eval.csv",
    output_csv="data/eval_labeled.csv",
    log_jsonl="logs/eval_labeling.jsonl"
)


run_experiment_0(
    input_path="data/test.csv",
    output_csv="data/test_labeled.csv",
    log_jsonl="logs/test_labeling.jsonl"
)