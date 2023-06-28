import random
from typing import Optional

import torch
from collections import deque

class ReplayMemory:
    def __init__(self, max_size=10000) -> None:
        self.max_size = max_size
        self.memory = deque([], maxlen=max_size)
        pass

    def sample(self, num_samples): 
        return random.sample(self.memory, num_samples)
    
    def insert(self, sample):
        self.memory.append(sample)

    def size(self):
        return len(self.memory)
    
class GameReplayMemory(ReplayMemory):
    def __init__(self, max_size=10000) -> None:
        super().__init__(max_size)
    
    def sample(self, num_samples):
        games = random.choices(self.memory, k=num_samples)
        samples = []
        for game in games:
            samples.append(random.sample(game, 1)[0])
        return samples
    
class EpisodeMemory:
    def __init__(self, num_parallel_envs: int, device: torch.device) -> None:
        self.memory = [[] for _ in range(num_parallel_envs)]
        self.num_parallel_envs = num_parallel_envs
        self.device = device

    def insert(self, inputs: torch.Tensor, action_visits: torch.Tensor):
        inputs = inputs.clone().to(device=self.device)
        action_visits = action_visits.clone().to(device=self.device)

        for i in range(self.num_parallel_envs):
            self.memory[i].append((inputs[i], action_visits[i]))

    def pop_terminated_episodes(self, terminated: torch.Tensor):
        episodes = []
        for i in range(self.num_parallel_envs):
            if terminated[i]:
                episode = self.memory[i]
                episodes.append(episode)
                self.memory[i] = []
        return episodes
    


