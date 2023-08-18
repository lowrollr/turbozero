




from dataclasses import dataclass
import torch
from core.algorithms.baselines.baseline import Baseline
from core.algorithms.evaluator import EvaluatorConfig
from core.env import Env

@dataclass
class GreedyConfig(EvaluatorConfig):
    heuristic: str


class GreedyBaseline(Baseline):
    def __init__(self, env: Env, config: GreedyConfig, *args, **kwargs):
        super().__init__(env, config, *args, **kwargs)
        self.metrics_key = 'greedy_' + config.heuristic
        self.proper_name = 'Greedy_' + config.heuristic
        self.config: GreedyConfig

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
            terminated = self.env.step(torch.full((self.env.parallel_envs, ), action_idx, dtype=torch.long, device=self.device, requires_grad=False))
            rewards = self.env.get_rewards(starting_players)
            greedy_rewards = self.env.get_greedy_rewards(starting_players)
            is_legal = legal_actions[:, action_idx]
            action_scores[:, action_idx] = ((rewards * terminated) + (greedy_rewards * (~terminated))) * is_legal

            self.env.load_node(torch.full_like(starting_players, True, dtype=torch.bool, device=self.device, requires_grad=False), saved=saved)

        return action_scores, (action_scores * legal_actions).max(dim=1)

        


