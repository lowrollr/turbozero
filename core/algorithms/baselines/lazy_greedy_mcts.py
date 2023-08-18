



from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from core.algorithms.baselines.baseline import Baseline
from core.algorithms.lazy_mcts import LazyMCTS, LazyMCTSConfig
from core.env import Env

@dataclass
class LazyGreedyMCTSConfig(LazyMCTSConfig):
    heuristic: str


class LazyGreedyMCTS(LazyMCTS, Baseline):
    def __init__(self, env: Env, config: LazyGreedyMCTSConfig, *args, **kwargs) -> None:
        super().__init__(env, config, *args, **kwargs)
        self.metrics_key = 'lazygreedymcts_' + config.heuristic + '_' + str(config.num_policy_rollouts)
        self.proper_name = 'LazyGreedyMCTS_' + config.heuristic + '_' + str(config.num_policy_rollouts)
        self.uniform_probabilities = torch.ones((self.env.parallel_envs, self.env.policy_shape[0]), device=self.device, requires_grad=False) / self.env.policy_shape[0]
        self.config: LazyGreedyMCTSConfig

    def evaluate(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        evaluation_fn = lambda env: (self.uniform_probabilities, env.get_greedy_rewards())
        return super().evaluate(evaluation_fn)

    
