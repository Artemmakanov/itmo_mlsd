import time
import numpy as np
import pickle
import json
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict

from research.src.inference import GuardrailInference


class GuardrailProfiler:
    def __init__(self, guardrail_instance):
        self.guardrail = guardrail_instance
        self.latencies: List[float] = []
        self.stage_stats = defaultdict(list)

    def benchmark(self, texts: List[str]):
        """
        Запуск пакетного теста для сбора статистики.
        """
        print(f"🚀 Starting benchmark on {len(texts)} samples...")
        
        for text in texts:
            start_time = time.perf_counter()
            
            # Вызов оригинального метода инференса
            result = self.guardrail.predict(text)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Сохраняем общую задержку
            self.latencies.append(latency_ms)
            
            # Сохраняем информацию по стадиям (если ваш predict возвращает метаданные)
            stage_info = "Stage 2" if result.get("score2") is not None else "Stage 1"
            self.stage_stats[stage_info].append(latency_ms)

    def print_summary(self):
        """
        Вывод итоговых метрик согласно дизайн-документу.
        """
        if not self.latencies:
            print("No data collected.")
            return

        p50 = np.percentile(self.latencies, 50)
        p95 = np.percentile(self.latencies, 95) # Ключевая метрика по БТ
        p99 = np.percentile(self.latencies, 99)
        avg = np.mean(self.latencies)
        rps = 1000 / avg if avg > 0 else 0

        print("\n" + "="*30)
        print("📊 INFERENCE PERFORMANCE")
        print("="*30)
        print(f"Total Requests: {len(self.latencies)}")
        print(f"Average Latency: {avg:.2f} ms")
        print(f"P50 (Median):    {p50:.2f} ms")
        print(f"**P95 (Target):   {p95:.2f} ms**") # Должно быть <= 100ms
        print(f"P99 (Tail):      {p99:.2f} ms")
        print(f"Throughput:      {rps:.2f} RPS")
        print("-" * 30)
        
        for stage, times in self.stage_stats.items():
            print(f"{stage} average: {np.mean(times):.2f} ms ({len(times)} calls)")


# Пример интеграции
if __name__ == "__main__":
    # 1. Инициализация (как в вашем коде)


# -------------------------
# Инициализация артефактов
# -------------------------
    with open("./results/mvp.json") as f:
        CONFIG = json.load(f)

    with open("./results/model1.pkl", "rb") as f:
        MODEL1 = pickle.load(f)

    try:
        with open("./results/model2.pkl", "rb") as f:
            MODEL2 = pickle.load(f)
    except FileNotFoundError:
        MODEL2 = None

    # Загрузка вспомогательных моделей для фичей
    CENTROIDS = None
    if "centroid_margin" in CONFIG["stage1_best"]["features"]:
        CENTROIDS = np.load("./results/centroids.npy", allow_pickle=True).item()

    ISO_MODEL = None
    if "anomaly_score" in CONFIG["stage1_best"]["features"]:
        with open("./results/iso_model.pkl", "rb") as f:
            ISO_MODEL = pickle.load(f)

    # Создание синглтона инференса
    guardrail = GuardrailInference(MODEL1, MODEL2, CONFIG, CENTROIDS, ISO_MODEL)
    
    # 2. Создание профилировщика
    profiler = GuardrailProfiler(guardrail)
    
    # 3. Набор тестовых данных (из примеров в дизайн-доке)
    test_prompts = pd.read_csv("./data/train.csv")['text'].iloc[:1000]

    # 4. Запуск и вывод
    profiler.benchmark(test_prompts)
    profiler.print_summary()