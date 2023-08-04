
from dataclasses import dataclass
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
        return torch.multinomial(torch.pow(visits, 1/self.config.temperature), 1, replacement=True).flatten()