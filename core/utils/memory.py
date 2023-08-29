import random
from typing import Optional
from core.utils.utils import cosine_centroid_similarity

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
    
    def similarity(self):
        episodes = torch.stack([x[0] for x in self.sample(4096)])
        return cosine_centroid_similarity(episodes)
            
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
    def __init__(self, parallel_envs: int, device: torch.device) -> None:
        self.memory = [[] for _ in range(parallel_envs)]
        self.parallel_envs = parallel_envs
        self.device = device

    def insert(self, inputs: torch.Tensor, action_visits: torch.Tensor, legal_actions: torch.Tensor):
        inputs = inputs.clone().to(device=self.device)
        action_visits = action_visits.clone().to(device=self.device)
        legal_actions = legal_actions.clone().to(device=self.device)

        for i in range(self.parallel_envs):
            self.memory[i].append((inputs[i], action_visits[i], legal_actions[i]))

    def pop_terminated_episodes(self, terminated: torch.Tensor):
        episodes = []
        for i in terminated.nonzero().flatten():
            episode = self.memory[i]
            episodes.append(episode)
            self.memory[i] = []
        return episodes
    
