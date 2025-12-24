# gr_eval/prompt_generator.py

from gr_eval.llm import HFSmallLLM
from gr_eval.prompt_spec import PromptSpec
from gr_eval.meta_prompt import META_PROMPT


class PromptGenerator:
    def __init__(self, llm: HFSmallLLM):
        self.llm = llm

    def generate(self, spec: PromptSpec, few_shots: list[tuple[str, str]] | None = None) -> str:
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

        system_prompt = self.llm.generate(meta_prompt, max_new_tokens=512)

        if few_shots:
            examples = "\n\n".join(
                f"if user asks: {p}\nAssistant must {a}"
                for p, a in few_shots
            )
            system_prompt += "\n\n### Examples of correct behavior:\n" + examples

        return system_prompt
