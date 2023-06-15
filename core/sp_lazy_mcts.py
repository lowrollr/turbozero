
from core.lazy_mcts import VectorizedLazyMCTS
from .vectenv import VectEnv
import torch



# THIS CLASS RELIES ON REWARDS BEING STRICTLY POSITIVE OR ZERO (FOR NOW)

class SinglePlayerVectorizedLazyMCTS(VectorizedLazyMCTS):
    def __init__(self, env: VectEnv, puct_coeff: float, very_positive_value: float) -> None:
        super().__init__(env, puct_coeff, very_positive_value)
        
    def choose_action_with_puct(self, probs: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
        n_sum = torch.sum(self.visit_counts, dim=1, keepdim=True)

        q_values = torch.where(
            self.visit_counts > 0,
            self.action_scores / self.visit_counts,
            self.very_positive_value
        )

        puct_scores = q_values + (self.puct_coeff * probs * torch.sqrt(n_sum + 1) / (1 + self.visit_counts))

        # even with puct score of zero only a legal action will be chosen
        legal_action_scores = (puct_scores * legal_actions) - (1 * torch.logical_not(legal_actions))
        chosen_actions = torch.argmax(legal_action_scores, dim=1, keepdim=True)

        return chosen_actions.squeeze(1)

    def iterate(self, model: torch.nn.Module, depth: int) -> torch.Tensor:
        while depth > 0:
            with torch.no_grad():
                policy_logits, values = model(self.env.states)
            
            legal_actions = self.env.get_legal_actions()
            distribution = torch.nn.functional.softmax(policy_logits, dim=1) * legal_actions
            next_actions = self.env.fast_weighted_sample(distribution, norm=True)
            depth -= 1
            if depth == 0:
                values.clamp_(0)
                return values.squeeze(1) * torch.logical_not(self.env.invalid_mask)
            self.env.step(next_actions)

    def explore_for_iters(self, model: torch.nn.Module,iters: int, search_depth: int) -> torch.Tensor:
        legal_actions = self.env.get_legal_actions()
        with torch.no_grad():
            policy_logits, _ = model(self.env.states)
        policy_logits = torch.nn.functional.softmax(policy_logits, dim=1)
        initial_state = self.env.states.clone()

        for _ in range(iters):
            actions = self.choose_action_with_puct(policy_logits, legal_actions)    
            self.env.step(actions)
            
            self.visit_counts[self.env.env_indices, actions] += 1
            self.action_scores[self.env.env_indices, actions] += self.iterate(model, search_depth)
            self.env.states = initial_state.clone()
            self.env.update_invalid_mask()

        return self.visit_counts