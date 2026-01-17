import pandas as pd
from sentence_transformers import SentenceTransformer

# --------------------------
# Загрузка и фильтр данных
# --------------------------
def load_data(filename):
    path_base = f"./data/{filename}.csv"
    path_labeled = f"./data/{filename}_labeled.csv"
    df_labeled = pd.read_csv(path_labeled)
    df_base = pd.read_csv(path_base)
    df_labeled =df_labeled[df_labeled["segment"].isin([
        "Blind Spot",
        "Naturally Allowed",
        "Obviously Refused"
    ])]
    # Paranoid Refusal убираем - их мало и они есть следствие ограниченности агента
    return df_base.merge(df_labeled.segment, left_index=True, right_index=True, how='inner')

# --------------------------
# Эмбеддинги через MiniLM
# --------------------------
class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.encoder.eval()

    def encode(self, texts):
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return embeddings
