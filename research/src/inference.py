import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from research.src.utils import Embedder
from research.src.data import build_features

class GuardrailInference:
    def __init__(
        self, 
        model1, 
        model2, 
        config: Dict[str, Any], 
        centroids: Optional[Any] = None, 
        iso_model: Optional[Any] = None
    ):
        """
        Инициализация каскадной системы защиты.
        """
        self.model1 = model1
        self.model2 = model2
        self.embedder = Embedder()
        self.centroids = centroids
        self.iso_model = iso_model
        
        # Загрузка конфигурации из дизайна
        self.stage1_cfg = config["stage1_best"]
        self.stage2_cfg = config["stage2"]
        
        self.features = self.stage1_cfg["features"]
        self.t1 = float(self.stage1_cfg["threshold"]) # Порог T1 = 0.24
        
        self.stage2_needed = self.stage2_cfg["needed"]
        # Порог T2 = 0.875
        self.t2 = float(self.stage2_cfg["thresholds"][1]) if self.stage2_needed else None

    def _score_model(self, model, X: pd.DataFrame) -> np.ndarray:
        """Вспомогательный метод для получения скора (вероятности или decision_function)."""
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        return model.predict_proba(X)[:, 1]

    def _extract_features(self, text: str) -> pd.DataFrame:
        """Эмбеддинг и формирование вектора признаков."""
        emb = self.embedder.encode([text])[0]
        sample = pd.DataFrame({
            "text": [text],
            "embedding": [emb]
        })
        return build_features(
            sample,
            self.features,
            self.centroids,
            self.iso_model
        )

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Основной цикл двустадийной фильтрации.
        """
        X = self._extract_features(text)

        # --- Stage 1 (SVC) ---
        score1 = float(self._score_model(self.model1, X)[0])
        # Если скор ниже порога T1, промпт считается безопасным
        if score1 < self.t1:
            return {
                "decision": "✅ ALLOW",
                "score1": score1,
                "score2": None,
                "t1": self.t1,
                "t2": self.t2
            }

        # --- Stage 2 (KNN) ---
        if self.stage2_needed and self.model2 is not None:
            score2 = float(self._score_model(self.model2, X)[0])
            # Блокировка происходит, если Stage 2 подтверждает атаку (score >= T2)
            is_attack = score2 >= self.t2
            
            return {
                "decision": "❌ REJECT (Stage 2)" if is_attack else "✅ ALLOW (Refined)",
                "score1": score1,
                "score2": score2,
                "t1": self.t1,
                "t2": self.t2
            }

        # Если Stage 2 не активен, решение принимается по Stage 1
        return {
            "decision": "❌ REJECT (Stage 1)",
            "score1": score1,
            "score2": None,
            "t1": self.t1,
            "t2": None
        }