
import torch

class OthelloVectEnv:
    def __init__(self, num_parallel_envs, device, progression_batch_size) -> None:
        self.num_parallel_envs = num_parallel_envs
        self.device = device

        self.boards = torch.zeros((num_parallel_envs, 1, 2, 8, 8), dtype=torch.float32, device=device)
        
        self.invalid_mask = torch.zeros((num_parallel_envs, 1, 1, 1, 1), dtype=torch.bool, device=device)

        