
import torch

from core import GLOB_FLOAT_TYPE
from core.vectenv import VectEnv
from .torchscripts import get_legal_actions, push_actions, build_filters, build_flips

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
        self.cur_player = torch.zeros((num_parallel_envs, ), dtype=torch.long, device=device, requires_grad=False)
        
        self.filters_and_indices = build_filters(device, board_size)
        self.flips = build_flips(num_rays, board_size, device)

        self.consecutive_passes = torch.zeros((num_parallel_envs, ), dtype=torch.long, device=device, requires_grad=False)

        self.need_to_calculate_rays = True
        self.reset()
        self.save_node()

        if debug:
            self.get_legal_actions_traced = get_legal_actions
            self.push_actions_traced = push_actions
        else:
            self.get_legal_actions_traced = torch.jit.trace(get_legal_actions, (self.states, self.ray_tensor, *self.filters_and_indices))
            self.push_actions_traced = torch.jit.trace(push_actions, (self.states, self.ray_tensor, torch.zeros((self.num_parallel_envs, ), dtype=torch.long, device=device), self.flips))

    def get_legal_actions(self):
        # adjacent to enemy tiles
        if self.need_to_calculate_rays:
            self.need_to_calculate_rays = False
            return self.get_legal_actions_traced(self.states, self.ray_tensor, *self.filters_and_indices)
        else:
            return self.ray_tensor.any(dim=1).view(-1, self.board_size ** 2)
    
    def push_actions(self, actions):
        if self.need_to_calculate_rays:
            self.need_to_calculate_rays = False
            self.get_legal_actions_traced(self.states, self.ray_tensor, *self.filters_and_indices)
        _, passes = self.push_actions_traced(self.states, self.ray_tensor, actions, self.flips)
        self.consecutive_passes += passes
        self.consecutive_passes *= passes
        self.need_to_calculate_rays = True
    
    def next_turn(self):
        self.states = torch.roll(self.states, 1, dims=1)
        self.cur_player += 1
        self.cur_player %= 2
    
    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.states.zero_()
        self.ray_tensor.zero_()
        self.rewards.zero_()
        self.terminated.zero_()
        self.cur_player.zero_()
        self.consecutive_passes.zero_()
        self.states[:, 0, 3, 3] = 1
        self.states[:, 1, 3, 4] = 1
        self.states[:, 1, 4, 3] = 1
        self.states[:, 0, 4, 4] = 1
        self.need_to_calculate_rays = True

    def is_terminal(self):
        return (self.states.sum(dim=(1, 2, 3)) == (self.board_size ** 2)) | self.consecutive_passes >= 2
    
    def update_terminated(self):
        super().update_terminated()
    
    def get_rewards(self):
        self.rewards.zero_()
        p1_sum = self.states[self.env_indices, self.cur_player].sum(dim=(1, 2))
        p2_sum = self.states[self.env_indices, (self.cur_player + 1) % 2].sum(dim=(1, 2))
        self.rewards += 1 * (p1_sum > p2_sum)
        self.rewards += 0.5 * (p1_sum == p2_sum)
        return self.rewards

    def reset_terminated_states(self):
        self.states *= 1 * ~self.terminated.view(-1, 1, 1, 1)
        self.cur_player *= 1 * ~self.terminated
        self.consecutive_passes *= 1 * ~self.terminated
        mask = 1 * self.terminated
        self.states[:, 0, 3, 3] += mask
        self.states[:, 1, 3, 4] += mask
        self.states[:, 1, 4, 3] += mask
        self.states[:, 0, 4, 4] += mask
        self.terminated.zero_()
        self.need_to_calculate_rays = True

    def save_node(self):
        self.saved = [
            self.states.clone(),
            self.cur_player.clone(),
            self.consecutive_passes.clone()
        ]
        
    def load_node(self, load_envs: torch.Tensor):
        load_envs_expnd = load_envs.view(-1, 1, 1, 1)
        self.states = self.saved[0].clone() * load_envs_expnd + self.states * (~load_envs_expnd)
        self.cur_player = self.saved[1].clone() * load_envs + self.cur_player * (~load_envs)
        self.consecutive_passes = self.saved[2].clone() * load_envs + self.consecutive_passes * (~load_envs)
        self.need_to_calculate_rays = True
        


