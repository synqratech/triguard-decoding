from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class TriGuardConfig:
    alpha: float = 0.06   # Entropy weight
    beta: float = 0.00    # (1-p_max) weight
    gamma: float = 0.00   # Margin weight
    delta: float = 0.75   # JSD weight
    eps: float = 0.46     # Critic weight
    zeta: float = 0.00    # Tail-JSD weight
    theta: float = 0.71   # Threshold
    t_cold: float = 0.2
    t_hot: float = 1.0
    t_metric: float = 1.0

PRESETS: Dict[str, TriGuardConfig] = {
    "balanced": TriGuardConfig(
        delta=0.75, eps=0.46, zeta=0.0, theta=0.71
    ),
    "tail_only": TriGuardConfig(
        delta=0.0, eps=0.0, zeta=1.0, alpha=0.0, theta=0.45,
        t_metric=1.0
    ),
    "strict": TriGuardConfig(
        delta=0.75, eps=0.5, zeta=0.5, theta=0.6
    )
}

def get_config(name: str) -> TriGuardConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
