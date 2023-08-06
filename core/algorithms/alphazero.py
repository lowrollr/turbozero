


from dataclasses import dataclass
import torch
from core.algorithms.evaluator import Evaluator, EvaluatorConfig
from core.algorithms.mcts import MCTS, MCTSConfig
from core.env import Env

@dataclass
class AlphaZeroConfig(MCTSConfig):
    temperature: float = 1.0


class AlphaZero(MCTS):
    def __init__(self, env: Env, config: AlphaZeroConfig) -> None:
        super().__init__(env, config)
        self.config: AlphaZeroConfig

    # all additional alphazero implementation details live in MCTS, for now
    def choose_actions(self, visits: torch.Tensor) -> torch.Tensor:
        if self.config.temperature > 0:
            return torch.multinomial(torch.pow(visits, 1/self.config.temperature), 1, replacement=True).flatten()
        else:
            return torch.argmax(visits, dim=1).flatten()
    

