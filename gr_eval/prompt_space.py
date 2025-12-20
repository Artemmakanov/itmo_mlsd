# gr_eval/prompt_sampling.py
from gr_eval.prompt_spec import PromptSpec

def sample_prompt_spec(trial) -> PromptSpec:
    return PromptSpec(
        # ── Style / persona ─────────────────────────────
        tone=trial.suggest_categorical(
            "tone",
            ["neutral", "strict", "calm", "educational", "cold"]
        ),
        role=trial.suggest_categorical(
            "role",
            ["policy_enforcer", "expert", "assistant", "reviewer"]
        ),

        # ── Core guardrail behavior ──────────────────────
        strictness=trial.suggest_categorical(
            "strictness",
            ["lenient", "balanced", "strict", "paranoid"]
        ),
        safety_focus=trial.suggest_categorical(
            "safety_focus",
            ["content_based", "intent_based", "capability_based"]
        ),
        intent_granularity=trial.suggest_categorical(
            "intent_granularity",
            ["binary", "multi_stage", "risk_score"]
        ),

        # ── Output format ────────────────────────────────
        verbosity=trial.suggest_int(
            "verbosity", 1, 5
        ),
        refusal_style=trial.suggest_categorical(
            "refusal_style",
            ["hard", "soft", "redirect", "explain"]
        ),
        reasoning=trial.suggest_categorical(
            "reasoning",
            ["none", "brief", "structured"]
        ),
        policy_explicitness=trial.suggest_categorical(
            "policy_explicitness",
            ["implicit", "semi_explicit", "explicit"]
        ),
        uncertainty_handling=trial.suggest_categorical(
            "uncertainty_handling",
            ["allow", "refuse", "ask_clarification"]
        ),
    )
