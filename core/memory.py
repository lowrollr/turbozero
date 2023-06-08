import random

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