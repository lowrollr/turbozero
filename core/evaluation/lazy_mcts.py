
import torch

from core.evaluation.evaluator import Evaluator
from core.evaluation.lazy_mcts_hypers import LazyMCTSHypers
from ..vectenv import VectEnv


class VectorizedLazyMCTS(Evaluator):
    def __init__(self, env: VectEnv, hypers: LazyMCTSHypers) -> None:
        super().__init__(env, env.device, hypers)

        self.action_scores = torch.zeros(
            (self.env.num_parallel_envs, *self.env.policy_shape),
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

        self.puct_coeff = hypers.puct_coeff
        self.policy_rollouts = hypers.num_policy_rollouts
        self.rollout_depth = hypers.rollout_depth

        # TODO: this is not an elegant solution at all but we need a large, positive, non-infinite value
        # this requires implementations to think carefully about choosing this value, which should be unnecessary
        self.very_positive_value = env.very_positive_value

        self.all_nodes = torch.ones(env.num_parallel_envs, dtype=torch.bool, device=self.device)

    def reset(self, seed=None) -> None:
        self.env.reset(seed=seed)
        self.reset_puct()

    def reset_puct(self) -> None:
        self.action_scores.zero_()
        self.visit_counts.zero_()

    def evaluate(self, model: torch.nn.Module) -> torch.Tensor:
        self.reset_puct()

        return self.explore_for_iters(model, self.policy_rollouts, self.rollout_depth)

    def choose_action_with_puct(self, probs: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
        n_sum = torch.sum(self.visit_counts, dim=1, keepdim=True)
        zero_counts = torch.logical_not(self.visit_counts)
        visit_counts_augmented = self.visit_counts + zero_counts
        q_values = self.action_scores / visit_counts_augmented
        q_values += self.very_positive_value * zero_counts

        puct_scores = q_values + \
            (self.puct_coeff * probs * torch.sqrt(n_sum + 1) / (1 + self.visit_counts))

        # even with puct score of zero only a legal action will be chosen
        legal_action_scores = (puct_scores * legal_actions) - \
            (self.very_positive_value * torch.logical_not(legal_actions))
        chosen_actions = torch.argmax(legal_action_scores, dim=1, keepdim=True)

        return chosen_actions.squeeze(1)

    def iterate(self, model: torch.nn.Module, depth: int, rewards: torch.Tensor) -> torch.Tensor:  # type: ignore
        while depth > 0:
            with torch.no_grad():
                policy_logits, values = model(self.env.states)
            depth -= 1
            if depth == 0:
                rewards = self.env.get_rewards()
                final_values = values.squeeze(
                    1) * torch.logical_not(self.env.terminated)
                final_values += rewards * self.env.terminated
                return final_values
            else:
                legal_actions = self.env.get_legal_actions()
                distribution = torch.nn.functional.softmax(
                    policy_logits, dim=1) * legal_actions
                next_actions = torch.multinomial(distribution, 1, replacement=True).flatten()
                self.env.step(next_actions)

    def explore_for_iters(self, model: torch.nn.Module, iters: int, search_depth: int) -> torch.Tensor:
        legal_actions = self.env.get_legal_actions()
        with torch.no_grad():
            policy_logits, _ = model(self.env.states)
        policy_logits = torch.nn.functional.softmax(policy_logits, dim=1)
        self.env.save_node()

        for _ in range(iters):
            actions = self.choose_action_with_puct(
                policy_logits, legal_actions)
            self.env.step(actions)
            rewards = self.env.get_rewards()
            values = self.iterate(model, search_depth, rewards)
            if search_depth % self.env.num_players:
                values = 1 - values
            self.visit_counts[self.env.env_indices, actions] += 1
            self.action_scores[self.env.env_indices, actions] += values
            self.env.load_node(self.all_nodes)

        return self.visit_counts
