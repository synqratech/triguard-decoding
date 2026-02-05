from .processor import TriGuardLogitsProcessor, TriGuardState, StopOnFallbackDone
from .presets import PRESETS, get_config
from transformers import LogitsProcessorList, StoppingCriteriaList

def build_triguard(
    tokenizer,
    preset: str = "balanced",
    fallback_text: str = "I am not sure. Please provide more context.",
    facts_mode: bool = True,
    prompt: str = None,
    **func_kwargs
):
    """
    Factory function to build Tri-Guard processors.
    
    Args:
        tokenizer: The tokenizer for the model.
        preset: 'balanced', 'tail_only', or 'strict'.
        fallback_text: Text to generate when guarding triggers.
        facts_mode: Whether to enable protection initially.
        prompt: The prompt text (for Critic).
        **func_kwargs: Overrides for specific weights (e.g. theta=0.5).
    """
    
    # 1. Load config
    config = get_config(preset)
    
    # 2. Override config with kwargs
    config_dict = config.__dict__.copy()
    for k, v in func_kwargs.items():
        if k in config_dict:
            config_dict[k] = v
            
    # 3. Initialize state
    fallback_ids = tokenizer(fallback_text, add_special_tokens=False).input_ids
    state = TriGuardState(fallback_ids=fallback_ids, facts_mode=facts_mode)
    
    # 4. Build Processor
    processor = TriGuardLogitsProcessor(
        tokenizer=tokenizer,
        state=state,
        alpha=config_dict["alpha"],
        beta=config_dict["beta"],
        gamma=config_dict["gamma"],
        delta=config_dict["delta"],
        eps=config_dict["eps"],
        zeta=config_dict["zeta"],
        theta=config_dict["theta"],
        t_cold=config_dict["t_cold"],
        t_hot=config_dict["t_hot"],
        t_metric=config_dict["t_metric"]
    )
    
    if prompt:
        processor.set_prompt(prompt)
        
    stopping_criteria = StopOnFallbackDone(state)
    
    return LogitsProcessorList([processor]), StoppingCriteriaList([stopping_criteria]), processor
