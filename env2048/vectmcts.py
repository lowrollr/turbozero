
import torch

from env2048.vectenv import Vectorized2048Env


class Vectorized2048MCTSLazy:
    def __init__(self, env, model, puct_c) -> None:
        self.env: Vectorized2048Env = env
        self.model = model
        self.inter_scores = torch.zeros((self.env.num_parallel_envs,), dtype=torch.float32, device=self.env.device, requires_grad=False)
        self.action_scores = torch.zeros((self.env.num_parallel_envs,4), dtype=torch.float32, device=self.env.device, requires_grad=False)
        self.visits = torch.zeros((self.env.num_parallel_envs,4), dtype=torch.float32, device=self.env.device, requires_grad=False)
        self.puct_c = puct_c
        self.indices = torch.arange(self.env.num_parallel_envs, dtype=torch.int64, device=self.env.device, requires_grad=False)
        self.very_positive_value = 1e5
    
    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.inter_scores.zero_()
        self.action_scores.zero_()
        self.visits.zero_()


    def choose_action_with_puct(self, probs, legal_actions):
        n_sum = torch.sum(self.visits, dim=1, keepdim=True)

        q_values = torch.where(self.visits > 0, self.action_scores / self.visits, self.very_positive_value)
        puct_scores = q_values + (self.puct_c * probs * torch.sqrt(n_sum + 1) / (1 + self.visits))
        legal_action_scores = puct_scores * legal_actions
        chosen_actions = torch.argmax(legal_action_scores, dim=1, keepdim=True)
        return chosen_actions.squeeze(1)

    def iterate(self, depth):
        d = depth
        while d > 0:
            with torch.no_grad():
                policy_logits, values = self.model(self.env.boards)
            values.clamp_(0)
            self.inter_scores += values.squeeze(1) * torch.logical_not(self.env.invalid_mask.view(self.env.num_parallel_envs))
            legal_actions = self.env.get_legal_moves()
            distribution = torch.nn.functional.softmax(policy_logits, dim=1) * legal_actions
            distribution.masked_fill_(distribution.sum(dim=1, keepdim=True) == 0, 1)    
            next_actions = torch.where(distribution.sum(dim=1) != 0, distribution.multinomial(1, True, generator=self.env.generator).squeeze(1), 0)
            self.env.step(next_actions)
            d -= 1
        
    def explore(self, iters, search_depth):
        self.action_scores.zero_()
        self.inter_scores.zero_()
        self.visits.zero_()
        legal_actions = self.env.get_legal_moves()
        with torch.no_grad():
            policy_logits, _ = self.model(self.env.boards)
        policy_logits = torch.nn.functional.softmax(policy_logits, dim=1)
        initial_state = self.env.boards.clone()
        for i in range(iters):
            actions = self.choose_action_with_puct(policy_logits, legal_actions)    
            self.env.step(actions)
            self.iterate(search_depth)
            self.visits[self.indices, actions] += 1
            self.action_scores[self.indices, actions] += self.inter_scores / search_depth
            self.inter_scores.zero_()
            self.env.boards = initial_state.clone()
            self.env.update_invalid_mask()

        return self.visits
    
    