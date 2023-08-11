
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from core.algorithms.evaluator import TrainableEvaluator


from core.algorithms.lazy_mcts import LazyMCTS, LazyMCTSConfig
from core.env import Env

@dataclass
class LazyZeroConfig(LazyMCTSConfig):
    temperature: float



class LazyZero(LazyMCTS, TrainableEvaluator):
    def __init__(self, env: Env, config: LazyZeroConfig, model: torch.nn.Module, *args, **kwargs):
        super().__init__(env, config, model, *args, **kwargs)
        self.config: LazyZeroConfig
        

    # all additional alphazero implementation details live in MCTS, for now
    def choose_actions(self, visits: torch.Tensor) -> torch.Tensor:
        if self.config.temperature > 0:
            return torch.multinomial(torch.pow(visits, 1/self.config.temperature), 1, replacement=True).flatten()
        else:
            return torch.argmax(visits, dim=1).flatten()
    
    def evaluate(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        evaluation_fn = lambda env: self.model(env.states)
        return super().evaluate(evaluation_fn)