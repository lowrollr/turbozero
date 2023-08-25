
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from core.algorithms.evaluator import TrainableEvaluator


from core.algorithms.lazy_mcts import LazyMCTS, LazyMCTSConfig
from core.env import Env
from core.utils.utils import rand_argmax_2d

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
            return rand_argmax_2d(visits).flatten()
    
    def evaluate(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        evaluation_fn = lambda env: self.model(env.get_nn_input())
        return super().evaluate(evaluation_fn)