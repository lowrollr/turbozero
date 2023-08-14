



from dataclasses import dataclass

import torch
from core.algorithms.baselines.baseline import Baseline
from core.algorithms.lazy_mcts import LazyMCTS, LazyMCTSConfig
from core.env import Env


@dataclass
class RandomRolloutLazyMCTSConfig(LazyMCTSConfig):
    rollouts_per_leaf: int

class RandomRolloutLazyMCTS(LazyMCTS, Baseline):
    def __init__(self, env: Env, config: RandomRolloutLazyMCTSConfig, *args, **kwargs):
        super().__init__(env, config, *args, **kwargs)
        self.config: RandomRolloutLazyMCTSConfig
        self.uniform_probabilities = torch.ones((self.env.parallel_envs, self.env.policy_shape[0]), device=self.device, requires_grad=False) / self.env.policy_shape[0]

    def evaluate(self):
        evaluation_fn = lambda env: (self.uniform_probabilities, env.random_rollout(num_rollouts=self.config.rollouts_per_leaf)) 
        return super().evaluate(evaluation_fn)