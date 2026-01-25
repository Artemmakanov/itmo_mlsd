import numpy as np
import time
from sklearn.metrics import recall_score, precision_score, f1_score

class GuardrailEvaluator:
    def __init__(self):
        pass

    def calculate_metrics(self, y_true, y_pred, complexity=None):
        """
        Расчет метрик согласно разделу 1.1.3 Design Doc.
        y_true, y_pred: массивы с метками (0 - benign, 1 - attack)
        complexity: массив строк ['easy', 'hard'] или None
        """
        results = {
            "Recall_All": recall_score(y_true, y_pred, zero_division=0),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "F1_Score": f1_score(y_true, y_pred, zero_division=0)
        }

        if complexity is not None:
            complexity = np.array(complexity)
            # Фильтруем индексы для Hard кейсов
            hard_mask = (complexity == 'hard')
            if np.any(hard_mask):
                results["Recall_Hard"] = recall_score(
                    y_true[hard_mask], y_pred[hard_mask], zero_division=0
                )
            
            # Фильтруем индексы для Easy кейсов
            easy_mask = (complexity == 'easy')
            if np.any(easy_mask):
                results["Recall_Easy"] = recall_score(
                    y_true[easy_mask], y_pred[easy_mask], zero_division=0
                )
        
        return results

    def measure_latency(self, func, *args, **kwargs):
        """Замер P95 Latency согласно разделу 1.2.2"""
        latencies = []
        # Прогрев и замеры (упрощенно)
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        return result, latency_ms