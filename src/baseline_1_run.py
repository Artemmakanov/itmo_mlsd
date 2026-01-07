import pandas as pd
import numpy as np
import logging
import sys
from src.models.knn_filter import KNNGuardrail
from src.metrics_factory import MetricsFactory

# Настройка форматированного вывода в CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s', # Упрощаем формат для лучшей читаемости табличных данных
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_baseline_1():
    # 1. Загрузка данных
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    b0_results = pd.read_csv("data/baseline_0_results.csv")
    
    # 2. Обучение KNN
    guardrail = KNNGuardrail(n_neighbors=5)
    guardrail.fit(train_df['text'].tolist(), train_df['label'].tolist())
    
    # 3. Получение вероятностей для всего теста
    probs = guardrail.predict_proba(test_df['text'].tolist())
    
    # 4. Калибровка порога (FPR 5%) по безопасным примерам
    benign_mask = (test_df['label'] == 0)
    threshold = np.percentile(probs[benign_mask], 95)
    
    # 5. Генерация графика распределения
    MetricsFactory.save_probability_distribution(
        probs=probs, 
        labels=test_df['label'].values, 
        threshold=threshold, 
        output_path="data/baseline_1_distribution.png"
    )
    
    # 6. Сбор данных для финальных метрик
    eval_df = pd.DataFrame({
        'label': test_df['label'],
        'refusal_decision': b0_results['refusal_decision'],
        'external_decision': probs >= threshold
    })

    # 7. Расчет и сохранение JSON метрик
    metrics = MetricsFactory.calculate_guardrail_metrics(
        eval_df, 
        output_json_path="data/baseline_1_metrics.json"
    )
    
    # Добавляем путь к графику в JSON лог
    metrics['artifacts'] = {"plot_path": "data/baseline_1_distribution.png"}
    
    logger.info(f"Baseline 1 Execution Finished. Recall on BS: {metrics['kpi']['recall_on_blind_spots']}")

if __name__ == "__main__":
    run_baseline_1()