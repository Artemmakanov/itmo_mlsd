import numpy as np
import pandas as pd
import re
from typing import List

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


def extract_pos_features(text: str) -> np.ndarray:
    if not isinstance(text, str) or len(text) == 0:
        return np.zeros(12)

    text_len = len(text)
    words = text.split()
    num_words = len(words)

    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0

    num_upper = sum(1 for c in text if c.isupper())
    num_digits = sum(1 for c in text if c.isdigit())
    num_special = sum(1 for c in text if not c.isalnum() and not c.isspace())

    frac_upper = num_upper / text_len
    frac_digits = num_digits / text_len
    frac_special = num_special / text_len

    num_q = text.count("?")
    num_exc = text.count("!")
    num_dot = text.count(".")

    START_VERB_PATTERN = re.compile(
        r"""^(
            please|pls|kindly|
            could\s+you|can\s+you|would\s+you|
            tell|give|show|explain|describe|list|summarize|outline|compare|analyze|
            write|generate|create|compose|draft|produce|build|design|
            modify|change|edit|improve|optimize|refactor|fix|convert|extract|
            pretend|act\s+as|assume|imagine|simulate|roleplay|
            implement|code|program|hack|bypass|exploit
        )\b
        """,
        re.IGNORECASE | re.VERBOSE
    )

    starts_with_verb = int(bool(START_VERB_PATTERN.match(text.strip())))
    starts_with_question = int(text.strip().endswith("?"))

    return np.array([
        text_len,
        num_words,
        avg_word_len,
        frac_upper,
        frac_digits,
        frac_special,
        num_q,
        num_exc,
        num_dot,
        starts_with_verb,
        starts_with_question,
        int(num_words < 5),  # очень короткий запрос
    ], dtype=float)


ATTACK_KEYWORDS = [
    "hack", "exploit", "bypass", "crack", "steal",
    "ddos", "malware", "phishing", "payload",
    "sql injection", "xss"
]

REFUSAL_KEYWORDS = [
    "i can't help", "i cannot assist", "not allowed",
    "policy", "i'm sorry", "cannot provide"
]

SAFETY_KEYWORDS = [
    "kill", "weapon", "bomb", "terrorist",
    "suicide", "drug", "poison"
]

JAILBREAK_KEYWORDS = [
    "ignore previous", "pretend to", "act as",
    "you are not bound", "system prompt"
]


def extract_keyword_flags(text: str) -> np.ndarray:
    if not isinstance(text, str):
        return np.zeros(6)

    text_l = text.lower()

    attack_flag = int(any(k in text_l for k in ATTACK_KEYWORDS))
    refusal_flag = int(any(k in text_l for k in REFUSAL_KEYWORDS))
    safety_flag = int(any(k in text_l for k in SAFETY_KEYWORDS))
    jailbreak_flag = int(any(k in text_l for k in JAILBREAK_KEYWORDS))

    num_attack_keywords = sum(text_l.count(k) for k in ATTACK_KEYWORDS)
    num_total_keywords = (
        num_attack_keywords
        + sum(text_l.count(k) for k in SAFETY_KEYWORDS)
        + sum(text_l.count(k) for k in JAILBREAK_KEYWORDS)
    )

    return np.array([
        attack_flag,
        refusal_flag,
        safety_flag,
        jailbreak_flag,
        num_attack_keywords,
        num_total_keywords
    ], dtype=float)



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

    if "pos" in features:
        scores = np.vstack(df["text"].apply(extract_pos_features).values)
        arrs.append(scores)

    if "keywords" in features:
        scores = np.vstack(df["text"].apply(extract_keyword_flags).values)
        arrs.append(scores)

    return np.hstack(arrs)