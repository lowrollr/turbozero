




from typing import Optional
import torch
from core.utils.memory import EpisodeMemory
from core.algorithms.evaluator import Evaluator


class Collector:
    def __init__(self, 
        evaluator: Evaluator, 
        episode_memory_device: torch.device
    ) -> None:
        parallel_envs = evaluator.env.parallel_envs
        self.evaluator = evaluator
        self.evaluator.reset()
        self.episode_memory = EpisodeMemory(parallel_envs, episode_memory_device)


    def collect(self, model: torch.nn.Module, inactive_mask: Optional[torch.Tensor] = None):
        terminated = self.collect_step(model).clone()

        if inactive_mask is not None:
            terminated *= ~inactive_mask
        terminated_episodes = self.episode_memory.pop_terminated_episodes(terminated)

        terminated_episodes = self.assign_rewards(terminated_episodes, terminated)

        self.evaluator.env.reset_terminated_states()

        return terminated_episodes, terminated
    
    def collect_step(self, model: torch.nn.Module):
        model.eval()
        initial_states, probs, _, _, terminated = self.evaluator.step(model)
        self.episode_memory.insert(initial_states, probs)
        return terminated
    
    def assign_rewards(self, terminated_episodes, terminated):
        raise NotImplementedError()
    
    def postprocess(self, terminated_episodes):
        return terminated_episodes
    
    def reset(self):
        self.episode_memory = EpisodeMemory(self.evaluator.env.parallel_envs, self.episode_memory.device)
        self.evaluator.reset()
