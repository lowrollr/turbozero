


import torch
from core.collector import Collector
from envs.othello.vectmcts import OthelloLazyMCTS


class OthelloCollector(Collector):
    def __init__(self,
        evaluator: OthelloLazyMCTS,
        episode_memory_device: torch.device,
        search_iters: int,
        search_depth: int
    ) -> None:
        super().__init__(evaluator, episode_memory_device, search_iters, search_depth)
        self.evaluator: OthelloLazyMCTS
        board_size = self.evaluator.env.board_size
        ids = torch.arange(self.evaluator.env.policy_shape[0])
        self.rotated_action_ids = torch.rot90(ids.reshape(board_size, board_size), k=1, dims=(0, 1)).flatten()

    def assign_rewards(self, terminated_episodes, terminated, info):
        episodes = []
        term_indices = terminated.nonzero(as_tuple=False).flatten()
        for i, episode in enumerate(terminated_episodes):
            episode_with_rewards = []
            ti = term_indices[i]
            r1, r2 = info['rewards'][ti].item(), 1 - info['rewards'][ti].item()
            p1_reward, p2_reward = (r2, r1) if len(episode) % 2 else (r1, r2)
            for ei, (inputs, visits) in enumerate(episode):
                episode_with_rewards.append((inputs, visits, torch.tensor(p2_reward if ei%2 else p1_reward, dtype=torch.float32, requires_grad=False, device=inputs.device)))
            episodes.append(episode_with_rewards)
        return episodes
    
    def postprocess(self, terminated_episodes):
        inputs, probs, rewards = zip(*terminated_episodes)
        rotated_inputs = []
        for i in inputs:
            for k in range(4):
                rotated_inputs.append(torch.rot90(i, k=k, dims=(1, 2)))
        rotated_probs = []
        for p in probs:
            # left -> down
            # down -> right
            # right -> up
            # up -> left
            new_p = p
            for k in range(4):
                rotated_probs.append(new_p)
                new_p = new_p[self.rotated_action_ids]

        rotated_rewards = []
        for r in rewards:
            rotated_rewards.extend([r] * 4)
        
        return list(zip(rotated_inputs, rotated_probs, rotated_rewards))