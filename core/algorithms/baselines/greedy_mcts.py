



from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from core.algorithms.baselines.baseline import Baseline
from core.algorithms.evaluator import Evaluator
from core.algorithms.mcts import MCTS, MCTSConfig
from core.env import Env

@dataclass
class GreedyMCTSConfig(MCTSConfig):
    heuristic: str


class GreedyMCTS(MCTS, Baseline):
    def __init__(self, env: Env, config: GreedyMCTSConfig, *args, **kwargs) -> None:
        super().__init__(env, config, *args, **kwargs)
        self.metrics_key = 'greedymcts_' + config.heuristic + '_' + str(config.num_iters)
        self.proper_name = 'GreedyMCTS_' + config.heuristic + '_' + str(config.num_iters)
        self.uniform_probabilities = torch.ones((self.env.parallel_envs, self.env.policy_shape[0]), device=self.device, requires_grad=False) / self.env.policy_shape[0]
        self.config: GreedyMCTSConfig

    def evaluate(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        evaluation_fn = lambda env: (self.uniform_probabilities, env.get_greedy_rewards())
        return super().evaluate(evaluation_fn)

    
