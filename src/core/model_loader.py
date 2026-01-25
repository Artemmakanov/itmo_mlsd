import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

class ModelLoader:
    def __init__(self, target_model_name="Qwen/Qwen2.5-1.5B-Instruct", embedder_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Инициализация загрузчика моделей.
        Согласно дизайну используем Qwen 2.5 1.5B и MiniLM.
        """
        self.target_model_name = target_model_name
        self.embedder_name = embedder_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_target_llm(self):
        """
        Загрузка основной защищаемой модели (Target LLM).
        Используется bfloat16 для оптимизации памяти на 1080 Ti.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name,
            torch_dtype="auto",
            device_map="auto" # Автоматически распределит на доступные GPU
        )
        
        # Обертка для удобного вызова генерации и проверки Self-refusal
        class TargetModelWrapper:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
                self.refusal_keywords = [
                    "i'm not able",
                    "sorry",
                    "not allowed",
                    "cannot",
                    "can't",
                    "against policy",
                ]

            def generate(self, prompt, max_new_tokens=64):
                # 1. Токенизируем вход
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                input_ids = inputs["input_ids"]
                
                # 2. Генерируем ответ
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id # Хорошая практика для избежания ворнингов
                )
                
                # 3. Срезаем входные токены из результата
                # outputs[0] содержит [input_ids + generated_ids]
                generated_tokens = outputs[0][len(input_ids[0]):]
                
                # 4. Декодируем только новые токены
                return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            def check_self_refusal(self, prompt):
                """Логика Baseline 0: проверка встроенного отказа"""
                response = self.generate(prompt)
                return any(kw.lower() in response.lower() for kw in self.refusal_keywords)

        return TargetModelWrapper(model, tokenizer)

    def load_embedding_model(self):
        """
        Загрузка эмбеддера для Baseline 1 (KNN).
        """
        model = SentenceTransformer(self.embedder_name, device=self.device)
        return model

    # модель для мета промпта - там max_new_tokens доожен быть большим!

# Утилитарные функции для быстрого доступа
def get_target_llm():
    return ModelLoader().load_target_llm()

def get_embedder():
    return ModelLoader().load_embedding_model()