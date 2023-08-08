
from dataclasses import dataclass
from typing import Optional, Tuple
import torch


from core.algorithms.lazy_mcts import LazyMCTS, LazyMCTSConfig
from core.env import Env

@dataclass
class LazyZeroConfig(LazyMCTSConfig):
    temperature: float



class LazyZero(LazyMCTS):
    def __init__(self, env: Env, config: LazyZeroConfig):
        super().__init__(env, config)
        self.config: LazyZeroConfig
        

    # all additional alphazero implementation details live in MCTS, for now
    def choose_actions(self, visits: torch.Tensor) -> torch.Tensor:
        if self.config.temperature > 0:
            return torch.multinomial(torch.pow(visits, 1/self.config.temperature), 1, replacement=True).flatten()
        else:
            return torch.argmax(visits, dim=1).flatten()
    
    def evaluate(self, model) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        evaluation_fn = lambda env: model(env.states)
        return super().evaluate(evaluation_fn)