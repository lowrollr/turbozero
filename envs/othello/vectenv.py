
import torch

from core import GLOB_FLOAT_TYPE
from core.vectenv import VectEnv
from .torchscripts import get_legal_actions, push_actions

class OthelloVectEnv(VectEnv):
    def __init__(self, 
        num_parallel_envs: int, 
        device: torch.device,
        board_size: int = 8,
        debug=False
    ) -> None:
        state_shape = (2, board_size, board_size)
        policy_shape = (64,)
        value_shape = (2, )
        super().__init__(num_parallel_envs, state_shape, policy_shape, value_shape, device)

        self.board_size = board_size
        num_rays = (8 * (self.board_size - 2)) + 1
        self.ray_tensor = torch.zeros((num_parallel_envs, num_rays, self.board_size, self.board_size), dtype=torch.float32, device=device, requires_grad=False)
        self.reset()

        if debug:
            self.get_legal_actions_traced = get_legal_actions
            self.push_actions_traced = push_actions
        else:
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
        self.rewards.zero_()
        self.terminated.zero_()
        self.states[:, 0, 3, 3] = 1
        self.states[:, 1, 3, 4] = 1
        self.states[:, 1, 4, 3] = 1
        self.states[:, 0, 4, 4] = 1

    def is_terminal(self):
        return self.states.sum(dim=(1, 2, 3)) == (self.board_size ** 2)
    
    def get_rewards(self):
        self.rewards.zero_()
        p1_sum = self.states[:, 0].sum(dim=(1, 2))
        p2_sum = self.states[:, 1].sum(dim=(1, 2))
        self.rewards += 1 * (p1_sum > p2_sum)
        self.rewards += 0.5 * (p1_sum == p2_sum)
        return self.rewards

    def reset_terminated_states(self):
        self.states *= 1 * ~self.terminated.view(-1, 1, 1, 1)
        self.states[:, 0, 3, 3] += 1 * self.terminated
        self.states[:, 1, 3, 4] += 1 * self.terminated
        self.states[:, 1, 4, 3] += 1 * self.terminated
        self.states[:, 0, 4, 4] += 1 * self.terminated
        self.terminated.zero_()
        

