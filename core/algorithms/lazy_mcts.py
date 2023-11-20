
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import torch

from core.algorithms.evaluator import Evaluator, EvaluatorConfig
from core.utils.utils import rand_argmax_2d
from ..env import Env


@dataclass 
class LazyMCTSConfig(EvaluatorConfig):
    num_policy_rollouts: int # number of policy rollouts to run per evaluation call
    rollout_depth: int # depth of each policy rollout, once this depth is reached, return the network's evaluation (value head) of the state
    puct_coeff: float # C-value in PUCT formula


class LazyMCTS(Evaluator):
    def __init__(self, env: Env, config: LazyMCTSConfig, *args, **kwargs) -> None:
        super().__init__(env, config, *args, **kwargs)

        self.action_scores = torch.zeros(
            (self.env.parallel_envs, *self.env.policy_shape),
            dtype=torch.float32,
            device=self.env.device,
            requires_grad=False
        )

        self.visit_counts = torch.zeros_like(
            self.action_scores,
            dtype=torch.float32,
            device=self.env.device,
            requires_grad=False
        )

        self.puct_coeff = config.puct_coeff
        self.policy_rollouts = config.num_policy_rollouts
        self.rollout_depth = config.rollout_depth

        self.all_nodes = torch.ones(env.parallel_envs, dtype=torch.bool, device=self.device)

    def reset(self, seed=None) -> None:
        self.env.reset(seed=seed)
        self.reset_puct()

    def reset_puct(self) -> None:
        self.action_scores.zero_()
        self.visit_counts.zero_()

    def evaluate(self, evaluation_fn: Callable) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.reset_puct()

        return self.explore_for_iters(evaluation_fn, self.policy_rollouts, self.rollout_depth)

    def choose_action_with_puct(self, probs: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
        n_sum = torch.sum(self.visit_counts, dim=1, keepdim=True)
        zero_counts = self.visit_counts == 0
        visit_counts_augmented = self.visit_counts + zero_counts
        q_values = self.action_scores / visit_counts_augmented

        puct_scores = q_values + \
            (self.puct_coeff * probs * torch.sqrt(n_sum + 1) / (1 + self.visit_counts))

        puct_scores = (puct_scores * legal_actions) + (torch.finfo(torch.float32).min * (~legal_actions))
        return torch.argmax(puct_scores, dim=1)
    
    def iterate(self, evaluation_fn: Callable, depth: int, rewards: torch.Tensor) -> torch.Tensor:  # type: ignore
        while depth > 0:
            with torch.no_grad():
                policy_logits, values = evaluation_fn(self.env)
            depth -= 1
            if depth == 0:
                rewards = self.env.get_rewards()
                final_values = values.flatten() * torch.logical_not(self.env.terminated)
                final_values += rewards * self.env.terminated
                return final_values
            else:
                legal_actions = self.env.get_legal_actions()
                policy_logits = (policy_logits * legal_actions) + (torch.finfo(torch.float32).min * (~legal_actions))
                distribution = torch.nn.functional.softmax(
                    policy_logits, dim=1)
                next_actions = torch.multinomial(distribution + self.env.is_terminal().unsqueeze(1), 1, replacement=True).flatten()
                self.env.step(next_actions)

    def explore_for_iters(self, evaluation_fn: Callable, iters: int, search_depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        legal_actions = self.env.get_legal_actions()
        with torch.no_grad():
            policy_logits, initial_values = evaluation_fn(self.env)
        policy_logits = (policy_logits * legal_actions) + (torch.finfo(torch.float32).min * (~legal_actions))
        policy_logits = torch.nn.functional.softmax(policy_logits, dim=1)
        saved = self.env.save_node()

        for _ in range(iters):
            actions = self.choose_action_with_puct(
                policy_logits, legal_actions)
            self.env.step(actions)
            rewards = self.env.get_rewards()
            values = self.iterate(evaluation_fn, search_depth, rewards)
            if search_depth % self.env.num_players:
                values = 1 - values
            self.visit_counts[self.env.env_indices, actions] += 1
            self.action_scores[self.env.env_indices, actions] += values
            self.env.load_node(self.all_nodes, saved)

        return self.visit_counts / self.visit_counts.sum(dim=1, keepdim=True), initial_values
