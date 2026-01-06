import pandas as pd
import json
import logging
import sys
from tqdm import tqdm
from datetime import datetime
from src.models.target_llm import TargetLLM

# Настройка форматированного вывода в CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s', # Упрощаем формат для лучшей читаемости табличных данных
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_experiment_0(input_path, output_csv, output_json):
    logger.info("="*80)
    logger.info(f"STARTING BASELINE 0 EVALUATION | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Failed to load data from {input_path}. Error: {e}")
        return

    logger.info(f"Target Model: Qwen 2.5 1.5B-Instruct")
    logger.info(f"Dataset Size: {len(df)} samples")
    logger.info("-"*80)

    llm = TargetLLM()
    full_log = []
    
    # Итерируемся по данным
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Prompts"):
        prompt_text = row['text']
        true_label = int(row['label']) # 1 - Attack, 0 - Benign
        
        # Инференс и проверка на отказ
        response_text = llm.generate(prompt_text)
        is_refused = llm.check_self_refusal(response_text)
        
        # Определение сегмента по матрице дизайна
        if true_label == 1:
            segment = "Obviously Refused" if is_refused else "Blind Spot"
        else:
            segment = "Naturally Allowed" if not is_refused else "Paranoid Refusal"
            
        # Формируем запись со всеми полями
        entry = {
            "id": idx,
            "text": prompt_text,            # Исходный промпт
            "output": response_text,        # Ответ модели
            "refusal_decision": is_refused, # Решение по refusal (True/False)
            "label": true_label,            # Метка (1/0)
            "segment": segment,             # Аналитический сегмент
            "timestamp": datetime.now().isoformat()
        }
        full_log.append(entry)
        
        # Расширенное логгирование в CLI для каждого кейса
        # Используем разные уровни логгирования для визуального выделения
        log_msg = (
            f"\n[ID: {idx} | Label: {true_label} | Segment: {segment}]\n"
            f"INPUT: {prompt_text[:100]}...\n"
            f"OUTPUT: {response_text[:100]}...\n"
            f"REFUSAL: {'YES' if is_refused else 'NO'}\n"
            f"{'-'*40}"
        )
        
        if segment == "Blind Spot":
            logger.warning(f"!!! ATTENTION: {log_msg}") # Выделяем критические пропуски атак
        elif segment == "Paranoid Refusal":
            logger.info(f"??? OVER-SAFETY: {log_msg}") # Выделяем ложные срабатывания
        else:
            # Для обычных кейсов пишем только в debug или просто пропускаем, 
            # чтобы не захламлять консоль (опционально)
            pass

    # Сохранение результатов
    res_df = pd.DataFrame(full_log)
    res_df.to_csv(output_csv, index=False)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(full_log, f, ensure_ascii=False, indent=4)
    
    # Финальный отчет в CLI
    logger.info("\n" + "="*80)
    logger.info("FINAL EXPERIMENT SUMMARY")
    logger.info("="*80)
    stats = res_df['segment'].value_counts()
    for seg_name, count in stats.items():
        percentage = (count / len(df)) * 100
        logger.info(f"{seg_name:<20}: {count:>4} ({percentage:>5.1f}%)")
    logger.info("="*80)
    logger.info(f"CSV Report: {output_csv}")
    logger.info(f"JSON Dump:   {output_json}")

if __name__ == "__main__":
    run_experiment_0(
        "data/test.csv", 
        "data/baseline_0_results.csv", 
        "data/target_llm_responses.json"
    )