



from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from core.algorithms.baselines.baseline import Baseline
from core.algorithms.evaluator import EvaluatorConfig
from core.algorithms.mcts import MCTS, MCTSConfig
from core.env import Env
from functools import partial

@dataclass
class RandomRolloutMCTSConfig(MCTSConfig):
    rollouts_per_leaf: int


    
class RandomRolloutMCTS(MCTS, Baseline):
    def __init__(self, env: Env, config: RandomRolloutMCTSConfig, *args, **kwargs):
        super().__init__(env, config, *args, **kwargs)
        self.config: RandomRolloutMCTSConfig
        self.uniform_probabilities = torch.ones((self.env.parallel_envs, self.env.policy_shape[0]), device=self.device, requires_grad=False) / self.env.policy_shape[0]


    def evaluate(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        evaluation_fn = lambda env: (self.uniform_probabilities, env.random_rollout(num_rollouts=self.config.rollouts_per_leaf)) 
        return super().evaluate(evaluation_fn)


        