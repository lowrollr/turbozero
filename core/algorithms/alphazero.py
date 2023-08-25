


from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import torch
from core.algorithms.evaluator import Evaluator, EvaluatorConfig, TrainableEvaluator
from core.algorithms.mcts import MCTS, MCTSConfig
from core.env import Env
from core.utils.utils import rand_argmax_2d

@dataclass
class AlphaZeroConfig(MCTSConfig):
    temperature: float = 1.0


class AlphaZero(MCTS, TrainableEvaluator):
    def __init__(self, env: Env, config: AlphaZeroConfig, model: torch.nn.Module) -> None:
        super().__init__(env, config, model)
        self.config: AlphaZeroConfig

    def choose_actions(self, visits: torch.Tensor) -> torch.Tensor:
        if self.config.temperature > 0:
            return torch.multinomial(torch.pow(visits, 1/self.config.temperature), 1, replacement=True).flatten()
        else:
            return rand_argmax_2d(visits).flatten()

    # see MCTS        
    def evaluate(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        evaluation_fn = lambda env: self.model(env.get_nn_input())
        return super().evaluate(evaluation_fn)
    

