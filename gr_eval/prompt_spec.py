from dataclasses import dataclass


@dataclass
class PromptSpec:
    tone: str
    role: str
    verbosity: int
    refusal_style: str
    reasoning: str
