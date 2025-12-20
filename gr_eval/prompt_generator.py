from gr_eval.llm import HFSmallLLM
from gr_eval.prompt_spec import PromptSpec
from gr_eval.meta_prompt import META_PROMPT


class PromptGenerator:
    def __init__(self, llm: HFSmallLLM):
        self.llm = llm

    def generate(self, spec: PromptSpec) -> str:
        meta_prompt = META_PROMPT.format(
            tone=spec.tone,
            role=spec.role,
            verbosity=spec.verbosity,
            refusal_style=spec.refusal_style,
            reasoning=spec.reasoning,
        )
        return self.llm.generate(meta_prompt)
