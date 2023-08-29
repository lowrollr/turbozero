
import torch
from core.algorithms.evaluator import TrainableEvaluator
from core.train.collector import Collector
from envs.othello.env import  OthelloEnvConfig


class OthelloCollector(Collector):
    def __init__(self,
        evaluator: TrainableEvaluator,
        episode_memory_device: torch.device
    ) -> None:
        super().__init__(evaluator, episode_memory_device)
        assert isinstance(evaluator.env.config, OthelloEnvConfig)
        board_size = self.evaluator.env.config.board_size
        ids = torch.arange(self.evaluator.env.policy_shape[0]-1, device=episode_memory_device)
        self.rotated_action_ids = torch.zeros((board_size**2)+1, dtype=torch.long, requires_grad=False, device=ids.device)
        self.rotated_action_ids[:board_size**2] = torch.rot90(ids.reshape(board_size, board_size), k=1, dims=(0, 1)).flatten()
        self.rotated_action_ids[board_size**2] = board_size**2

    def assign_rewards(self, terminated_episodes, terminated):
        episodes = []
        
        if terminated.any():
            term_indices = terminated.nonzero(as_tuple=False).flatten()
            rewards = self.evaluator.env.get_rewards(torch.zeros_like(self.evaluator.env.env_indices)).clone().cpu().numpy()
            for i, episode in enumerate(terminated_episodes):
                episode_with_rewards = []
                ti = term_indices[i]
                p1_reward = rewards[ti]
                p2_reward = 1 - p1_reward
                for ei, (inputs, visits, legal_actions) in enumerate(episode):
                    if visits.sum(): # only append states where a move was possible
                        episode_with_rewards.append((inputs, visits, torch.tensor(p2_reward if ei%2 else p1_reward, dtype=torch.float32, requires_grad=False, device=inputs.device), legal_actions))
                episodes.append(episode_with_rewards)
        return episodes
    
    def postprocess(self, terminated_episodes):
        inputs, probs, rewards, legal_actions = zip(*terminated_episodes)
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

        rotated_legal_actions = []
        for l in legal_actions:
            new_l = l
            for k in range(4):
                rotated_legal_actions.append(new_l)
                new_l = new_l[self.rotated_action_ids]

        rotated_rewards = []
        for r in rewards:
            rotated_rewards.extend([r] * 4)
        
        return list(zip(rotated_inputs, rotated_probs, rotated_rewards, rotated_legal_actions))