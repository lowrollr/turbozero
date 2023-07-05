from dataclasses import dataclass
from typing import Optional

@dataclass()
class LazyMCTSHypers:
    num_policy_rollouts: int
    rollout_depth: int
    puct_coeff: float = 1.0