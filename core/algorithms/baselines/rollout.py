





from dataclasses import dataclass
import torch
from core.algorithms.baselines.baseline import Baseline
from core.algorithms.evaluator import EvaluatorConfig
from core.env import Env

@dataclass
class RolloutConfig(EvaluatorConfig):
    num_rollouts: int

class Rollout(Baseline):
    def __init__(self, env: Env, config: RolloutConfig, *args, **kwargs):
        super().__init__(env, config, *args, **kwargs)
        self.config: RolloutConfig

    def evaluate(self):
        saved = self.env.save_node()
        legal_actions = self.env.get_legal_actions().clone()
        starting_players = self.env.cur_players.clone()
        action_scores = torch.zeros(
            (self.env.parallel_envs, self.env.policy_shape[0]),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        for action_idx in range(self.env.policy_shape[0]):
            rewards = self.env.random_rollout(num_rollouts=self.config.num_rollouts)
            is_legal = legal_actions[:, action_idx]
            action_scores[:, action_idx] = rewards * is_legal

            self.env.load_node(torch.full_like(starting_players, True, dtype=torch.bool, device=self.device, requires_grad=False), saved=saved)

        return action_scores, None