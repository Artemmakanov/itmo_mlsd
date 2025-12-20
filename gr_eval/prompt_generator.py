# gr_eval/prompt_generator.py

from gr_eval.llm import HFSmallLLM
from gr_eval.prompt_spec import PromptSpec
from gr_eval.meta_prompt import META_PROMPT


class PromptGenerator:
    def __init__(self, llm: HFSmallLLM):
        self.llm = llm

    def generate(self, spec: PromptSpec) -> str:
        meta_prompt = META_PROMPT.format(
            # persona / style
            tone=spec.tone,
            role=spec.role,

            # core guardrail behavior
            strictness=spec.strictness,
            safety_focus=spec.safety_focus,
            intent_granularity=spec.intent_granularity,

            # output / formatting
            verbosity=spec.verbosity,
            refusal_style=spec.refusal_style,
            reasoning=spec.reasoning,
            policy_explicitness=spec.policy_explicitness,
            uncertainty_handling=spec.uncertainty_handling,
        )

        return self.llm.generate(meta_prompt)
