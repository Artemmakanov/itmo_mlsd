# gr_eval/prompt_spec.py
from dataclasses import dataclass


@dataclass
class PromptSpec:
    # стиль и роль
    tone: str
    role: str

    # контроль поведения
    strictness: str
    safety_focus: str
    intent_granularity: str

    # формат ответа
    verbosity: int
    refusal_style: str
    reasoning: str
    policy_explicitness: str
    uncertainty_handling: str
