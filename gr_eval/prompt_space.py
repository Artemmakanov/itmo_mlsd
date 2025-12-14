from gr_eval.prompt_spec import PromptSpec

def sample_prompt_spec(trial) -> PromptSpec:
    return PromptSpec(
        tone=trial.suggest_categorical(
            "tone",
            ["neutral", "strict", "calm", "educational", "cold"]
        ),
        role=trial.suggest_categorical(
            "role",
            ["policy_enforcer", "expert", "assistant", "reviewer"]
        ),
        verbosity=trial.suggest_int("verbosity", 1, 5),
        refusal_style=trial.suggest_categorical(
            "refusal_style",
            ["hard", "soft", "redirect", "explain"]
        ),
        reasoning=trial.suggest_categorical(
            "reasoning",
            ["none", "brief", "structured"]
        ),
    )
