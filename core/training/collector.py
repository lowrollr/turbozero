




from typing import Optional
import torch
from core.utils.memory import EpisodeMemory
from core.evaluation.evaluator import Evaluator


class Collector:
    def __init__(self, 
        evaluator: Evaluator, 
        episode_memory_device: torch.device,
        temperature: Optional[float] = None
    ) -> None:
        num_parallel_envs = evaluator.env.num_parallel_envs
        self.evaluator = evaluator
        self.evaluator.reset()
        self.episode_memory = EpisodeMemory(num_parallel_envs, episode_memory_device)
        self.temperature = temperature


    def collect(self, model: torch.nn.Module, reset_terminal: bool = True):
        
        terminated = self.collect_step(model)

        terminated_episodes = self.episode_memory.pop_terminated_episodes(terminated)

        terminated_episodes = self.assign_rewards(terminated_episodes, terminated)

        if reset_terminal:
            self.evaluator.env.reset_terminated_states()

        return terminated_episodes, terminated
    
    def collect_step(self, model: torch.nn.Module):
        model.eval()

        visits = self.evaluator.evaluate(model)
        self.episode_memory.insert(self.evaluator.env.states, visits)
        if self.temperature is not None:
            actions = self.evaluator.env.fast_weighted_sample(torch.pow(visits, 1/self.temperature))
        else:
            actions = torch.argmax(visits, dim=1)

        terminated = self.evaluator.step_env(actions)
        return terminated
    
    def assign_rewards(self, terminated_episodes, terminated):
        raise NotImplementedError()
    
    def postprocess(self, terminated_episodes):
        return terminated_episodes
    
    def reset(self):
        self.episode_memory = EpisodeMemory(self.evaluator.env.num_parallel_envs, self.episode_memory.device)
        self.evaluator.reset()

    def get_details(self):
        return {
            'type': type(self.evaluator),
            'hypers': self.evaluator.hypers,
        }
