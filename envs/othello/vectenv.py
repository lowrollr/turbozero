
import torch

from core import GLOB_FLOAT_TYPE
from core.mp_vectenv import MPVectEnv
from .torchscripts import get_legal_actions, push_actions

class OthelloVectEnv(MPVectEnv):
    def __init__(self, 
        num_parallel_envs: int, 
        device: torch.device,
        board_size: int = 8   
    ) -> None:
        state_shape = (2, board_size, board_size)
        policy_shape = (64,)
        value_shape = (2, )
        super().__init__(2, num_parallel_envs, state_shape, policy_shape, value_shape, device, False)

        self.states = self.states.long()

        self.board_size = board_size
        num_rays = (8 * (self.board_size - 2)) + 1
        self.ray_tensor = torch.zeros((num_parallel_envs, num_rays, self.board_size, self.board_size), dtype=torch.long, device=device, requires_grad=False)
        self.reset()
        self.get_legal_actions_traced = torch.jit.trace(get_legal_actions, (self.states, self.ray_tensor))
        self.push_actions_traced = torch.jit.trace(push_actions, (self.states, self.ray_tensor, torch.zeros((self.num_parallel_envs, ), dtype=torch.long, device=device)))
    
    def get_legal_actions(self):
        # adjacent to enemy tiles
        return self.get_legal_actions_traced(self.states, self.ray_tensor)
    
    def push_actions(self, actions):
        self.push_actions_traced(self.states, self.ray_tensor, actions)
    
    def next_turn(self):
        self.states = torch.roll(self.states, 1, dims=1)
    
    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.states.zero_()
        self.ray_tensor.zero_()
        self.states[:, 0, 3, 3] = 1
        self.states[:, 1, 3, 4] = 1
        self.states[:, 1, 4, 3] = 1
        self.states[:, 0, 4, 4] = 1

    def is_terminal(self):
        return self.states.sum(dim=(1, 2, 3)) == (self.board_size ** 2)
    
    def get_rewards(self):
        p1_sum = self.states[:, 0].sum(dim=(1, 2))
        p2_sum = self.states[:, 1].sum(dim=(1, 2))
        return torch.stack([p1_sum > p2_sum, p2_sum > p1_sum], dim=1).float()

