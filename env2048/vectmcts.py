
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
    
    def choose_action_with_puct(self, probs, legal_actions):
        n_sum = torch.sum(self.visits, dim=1, keepdim=True)

        q_values = torch.where(self.visits > 0, self.action_scores / self.visits, 100)
        puct_scores = q_values + (self.puct_c * probs * torch.sqrt(n_sum + 1) / (1 + self.visits))
        legal_action_scores = puct_scores * legal_actions
        chosen_actions = torch.argmax(legal_action_scores, dim=1, keepdim=True)
        return chosen_actions.squeeze(1)

    def iterate(self, depth):
        d = depth
        while d > 0:
            with torch.no_grad():
                policy_logits, value = self.model(self.env.boards)
            policy_logits = torch.max(policy_logits, torch.tensor(0.00001))
            self.inter_scores += value.squeeze(1) * torch.logical_not(self.env.invalid_mask.view(self.env.num_parallel_envs))
            legal_actions = self.env.get_legal_moves()
            policy_logits *= legal_actions
            policy_logits.masked_fill_(policy_logits.amax(dim=1, keepdim=True) == 0, 1)
            next_actions = torch.multinomial(torch.nn.functional.softmax(policy_logits, dim=1), num_samples=1, replacement=True).squeeze(1)
            self.env.step(next_actions)
            d -= 1
        
    def explore(self, iters, search_depth):
        self.action_scores.fill_(0)
        self.inter_scores.fill_(0)
        self.visits.fill_(0)
        legal_actions = self.env.get_legal_moves()
        with torch.no_grad():
            policy_logits, _ = self.model(self.env.boards)
        policy_logits = torch.max(policy_logits, torch.tensor(0.00001))
        policy_logits = torch.nn.functional.softmax(policy_logits, dim=1)
        initial_state = self.env.boards.clone()
        for i in range(iters):
            actions = self.choose_action_with_puct(policy_logits, legal_actions)            
            self.env.step(actions)
            self.iterate(search_depth)
            self.visits[self.indices, actions] += 1
            self.action_scores[self.indices, actions] += self.inter_scores / search_depth
            self.inter_scores.fill_(0)
            self.env.boards = initial_state.clone()
            self.env.update_invalid_mask()

        return self.visits
    
    