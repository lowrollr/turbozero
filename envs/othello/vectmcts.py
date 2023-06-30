


import torch
from core.lazy_mcts import VectorizedLazyMCTS
from envs.othello.vectenv import OthelloVectEnv


class OthelloLazyMCTS(VectorizedLazyMCTS):
    def __init__(self, num_parallel_envs: int, device: torch.device, board_size: int, puct_c: float, debug: bool = False) -> None:
        env = OthelloVectEnv(num_parallel_envs, device, board_size=board_size, debug=debug)
        super().__init__(env, puct_c, 1e5)
        self.env: OthelloVectEnv

    def explore_for_iters(self, model: torch.nn.Module, iters: int, search_depth: int) -> torch.Tensor:
        legal_actions = self.env.get_legal_actions()
        env_ray_tensor, consecutive, initial_state, cur_player = self.env.ray_tensor.clone(), self.env.consecutive_passes.clone(), self.env.states.clone(), self.env.cur_player.clone()
        with torch.no_grad():
            policy_logits, _ = model(self.env.states)
        policy_logits = torch.nn.functional.softmax(policy_logits, dim=1)

        for _ in range(iters):
            actions = self.choose_action_with_puct(policy_logits, legal_actions)    
            self.env.step(actions)
            rewards = self.env.get_rewards()
            self.visit_counts[self.env.env_indices, actions] += 1
            
            value = self.iterate(model, search_depth, rewards)
            if search_depth % 2:
                value = 1 - value
            self.action_scores[self.env.env_indices, actions] += value
            self.env.states = initial_state.clone()
            self.env.ray_tensor = env_ray_tensor.clone()
            self.env.consecutive_passes = consecutive.clone()
            self.env.cur_player = cur_player.clone()
            self.env.update_terminated()

        
        return self.visit_counts