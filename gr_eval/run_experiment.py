import pandas as pd
import json
from tqdm import tqdm
from loguru import logger
from gr_eval.metrics import compute_metrics


RANDOM_STATE = 42

def score_llm_only(llm, df, verbose: bool = False):
    y_true, y_pred = [], []

    log = []

    for _, row in tqdm(df.iterrows(), total=len(df), disable=verbose):

        response = llm.generate(row["text"])
        refused = llm.refused(response)

        y_pred.append(1 if refused else 0)
        y_true.append(row["label"])

        record = {
            'input': row["text"],
            'response': response,
            'y_true': row["label"],
            'y_pred': 1 if refused else 0,
        }
        log.append(record)

        if verbose:
            text = (
                '\n'
                f"GroundTruth: {row['label']}, Prediction: {refused},\n"
                f"Input: {row["text"]},\nResponse from llm: {response}"
                '\n\n'
            )
            logger.info(text)

    with open("./log_score_llm_only.json", "w") as f:
        json.dump(log, f)

    return compute_metrics(y_true, y_pred)


def score_llm_with_knn_gr(llm, gr, df, verbose: bool = False):
    y_true, y_pred, y_score = [], [], []

    log = []

    probs = gr.predict_proba(df["text"].tolist())

    for i, row in tqdm(df.iterrows(), total=len(df), disable=verbose):
        if probs[i] > 0.5:
            response, refused = None, None
            y_score.append(probs[i])
            y_pred.append(1)
            pred = 1
        else:
            response = llm.generate(row.text)
            refused = llm.refused(response)

            y_score.append(probs[i])
            pred = 1 if refused else 0
            y_pred.append(pred)
            

        y_true.append(row.label)

        record = {
            'input': row["text"],
            'response': response,
            'y_true': row["label"],
            'y_pred': pred,
        }
        log.append(record)


        if verbose:
            if probs[i] > 0.5:
                text = (
                    '\n'
                    f"GroundTruth: {row['label']}, Prediction: {refused},\n"
                    f"Input: {row["text"]}\n"
                    '\n\n'
                )
            else:
                text = (
                    '\n'
                    f"GroundTruth: {row['label']}, Prediction: {refused},\n"
                    f"Input: {row["text"]},\nResponse from llm: {response}\n"
                    '\n\n'
                )
            logger.info(text)

    with open("./log_score_llm_with_knn_gr.json", "w") as f:
        json.dump(log, f)

    return compute_metrics(y_true, y_pred, y_score)


