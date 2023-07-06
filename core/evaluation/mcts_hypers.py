
from dataclasses import dataclass
from typing import Optional

@dataclass()
class MCTSHypers:
    num_iters: int
    max_depth: Optional[int] = None
    puct_coeff: float = 1.0
    dirichlet_alpha: float = 0.5
    dirichlet_epsilon: float = 0.25
