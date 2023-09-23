




import torch
from core.algorithms.evaluator import TrainableEvaluator
from core.train.collector import Collector
from envs.connect_x.env import ConnectXConfig


class ConnectXCollector(Collector):
    def __init__(self,
        evaluator: TrainableEvaluator,
        episode_memory_device: torch.device             
    ) -> None:
        super().__init__(evaluator, episode_memory_device)
        assert isinstance(evaluator.env.config, ConnectXConfig)
    
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