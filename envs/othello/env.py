
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

from core.env import Env, EnvConfig
from .torchscripts import get_legal_actions, push_actions, build_filters, build_flips

@dataclass
class OthelloEnvConfig(EnvConfig):
    board_size: int = 8





class OthelloEnv(Env):
    def __init__(self, 
        parallel_envs: int,
        config: OthelloEnvConfig,
        device: torch.device,
        debug=False
    ) -> None:
        self.board_size = config.board_size
        self.config: OthelloEnvConfig
        state_shape = torch.Size((2, self.board_size, self.board_size))
        policy_shape = torch.Size(((self.board_size ** 2) + 1,))
        value_shape = torch.Size((2, ))

        super().__init__(
            parallel_envs = parallel_envs,
            config = config,
            device=device,
            num_players=2, 
            state_shape=state_shape, 
            policy_shape=policy_shape, 
            value_shape=value_shape, 
            debug=debug
        )

        num_rays = (8 * (self.board_size - 2)) + 1
        self.ray_tensor = torch.zeros((self.parallel_envs, num_rays, self.board_size, self.board_size), dtype=torch.float32, device=device, requires_grad=False)
        
        self.filters_and_indices = build_filters(device, self.board_size)
        self.flips = build_flips(num_rays, self.board_size, device)

        self.consecutive_passes = torch.zeros((self.parallel_envs, ), dtype=torch.long, device=device, requires_grad=False)
        self.legal_actions = torch.zeros((self.parallel_envs, (self.board_size ** 2) + 1), dtype=torch.bool, device=device, requires_grad=False)

        self.need_to_calculate_rays = True
        self.reset()

        if self.debug:
            self.get_legal_actions_traced = get_legal_actions
            self.push_actions_traced = push_actions
        else:
            self.get_legal_actions_traced = torch.jit.trace(get_legal_actions, (self.states, self.ray_tensor, self.legal_actions, *self.filters_and_indices)) # type: ignore
            self.push_actions_traced = torch.jit.trace(push_actions, (self.states, self.ray_tensor, torch.zeros((self.parallel_envs, ), dtype=torch.long, device=device), self.flips)) # type: ignore

    def get_legal_actions(self):
        if self.need_to_calculate_rays:
            self.need_to_calculate_rays = False
            return self.get_legal_actions_traced(self.states, self.ray_tensor, self.legal_actions, *self.filters_and_indices) # type: ignore
        else:
            return self.legal_actions
    
    def push_actions(self, actions):
        if self.need_to_calculate_rays:
            self.get_legal_actions() # updates ray tensor
        _, passes = self.push_actions_traced(self.states, self.ray_tensor, actions, self.flips) # type: ignore
        self.consecutive_passes += passes
        self.consecutive_passes *= passes
        self.need_to_calculate_rays = True
    
    def next_turn(self):
        self.states = torch.roll(self.states, 1, dims=1)
        self.next_player()
    
    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.states.zero_()
        self.ray_tensor.zero_()
        self.rewards.zero_()
        self.terminated.zero_()
        self.cur_players.zero_()
        self.consecutive_passes.zero_()
        self.legal_actions.zero_()
        self.states[:, 0, 3, 3] = 1
        self.states[:, 1, 3, 4] = 1
        self.states[:, 1, 4, 3] = 1
        self.states[:, 0, 4, 4] = 1
        self.need_to_calculate_rays = True

    def is_terminal(self):
        return (self.states.sum(dim=(1, 2, 3)) == (self.board_size ** 2)) | (self.consecutive_passes >= 2)
    
    def update_terminated(self):
        super().update_terminated()
    
    def get_rewards(self, player_ids: Optional[torch.Tensor] = None):
        self.rewards.zero_()

        if player_ids is None:
            player_ids = self.cur_players
        idx = ((player_ids == self.cur_players).int() - 1) % 2
        other_idx = 1 - idx

        p1_sum = self.states[self.env_indices, idx].sum(dim=(1, 2))
        p2_sum = self.states[self.env_indices, other_idx].sum(dim=(1, 2))
        self.rewards += 1 * (p1_sum > p2_sum)
        self.rewards += 0.5 * (p1_sum == p2_sum)
        return self.rewards

    def reset_terminated_states(self):
        self.states *= 1 * ~self.terminated.view(-1, 1, 1, 1)
        self.cur_players *= 1 * ~self.terminated
        self.consecutive_passes *= 1 * ~self.terminated
        mask = 1 * self.terminated
        self.states[:, 0, 3, 3] += mask
        self.states[:, 1, 3, 4] += mask
        self.states[:, 1, 4, 3] += mask
        self.states[:, 0, 4, 4] += mask
        self.terminated.zero_()
        self.need_to_calculate_rays = True

    def save_node(self):
        return (
            self.states.clone(),
            self.cur_players.clone(),
            self.consecutive_passes.clone()
        )
        
    def load_node(self, load_envs: torch.Tensor, saved: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        load_envs_expnd = load_envs.view(-1, 1, 1, 1)
        self.states = saved[0].clone() * load_envs_expnd + self.states * (~load_envs_expnd)
        self.cur_players = saved[1].clone() * load_envs + self.cur_players * (~load_envs)
        self.consecutive_passes = saved[2].clone() * load_envs + self.consecutive_passes * (~load_envs)
        self.need_to_calculate_rays = True
        self.update_terminated()
    
    def get_greedy_rewards(self, player_ids: Optional[torch.Tensor] = None, heuristic: str = 'num_tiles'):
        if heuristic == 'minmax_moves':
            legal_actions_sum = self.get_legal_actions().sum(dim=1)
            return (((legal_actions_sum) * (self.cur_players == player_ids)) + (self.policy_shape[0] - legal_actions_sum) * (self.cur_players != player_ids)) / self.policy_shape[0]
        elif heuristic == 'num_tiles':
            if player_ids is None:
                player_ids = self.cur_players
            idx = ((player_ids == self.cur_players).int() - 1) % 2
            other_idx = 1 - idx 
            return 0.5 + ((self.states[self.env_indices, idx].sum(dim=(1, 2)) - self.states[self.env_indices, other_idx].sum(dim=(1, 2))) / (2 * (self.board_size ** 2)))
        elif heuristic == 'corners':
            if player_ids is None:
                player_ids = self.cur_players
            idx = ((player_ids == self.cur_players).int() - 1) % 2
            other_idx = 1 - idx 
            top_left_corner = self.states[self.env_indices, idx, 0, 0] - self.states[self.env_indices, other_idx, 0, 0]
            top_right_corner = self.states[self.env_indices, idx, 0, self.board_size - 1] - self.states[self.env_indices, other_idx, 0, self.board_size - 1]
            bottom_left_corner = self.states[self.env_indices, idx, self.board_size - 1, 0] - self.states[self.env_indices, other_idx, self.board_size - 1, 0]
            bottom_right_corner = self.states[self.env_indices, idx, self.board_size - 1, self.board_size - 1] - self.states[self.env_indices, other_idx, self.board_size - 1, self.board_size - 1]
            return 0.5 + ((top_left_corner + top_right_corner + bottom_left_corner + bottom_right_corner) / 8)
        elif heuristic == 'corners_and_edges':
            if player_ids is None:
                player_ids = self.cur_players
            idx = ((player_ids == self.cur_players).int() - 1) % 2
            other_idx = 1 - idx
            edge = self.states[self.env_indices, idx, 0, :] - self.states[self.env_indices, other_idx, 0, :]
            edge += self.states[self.env_indices, idx, :, 0] - self.states[self.env_indices, other_idx, :, 0]
            edge += self.states[self.env_indices, idx, :, self.board_size - 1] - self.states[self.env_indices, other_idx, :, self.board_size - 1]
            edge += self.states[self.env_indices, idx, self.board_size - 1, :] - self.states[self.env_indices, other_idx, self.board_size - 1, :]
            # corners are counted twice, but corners are good, so it's fine!
            circumference = (2 * self.board_size) + (2 * (self.board_size - 2))
            return 0.5 + (edge / (2 * circumference))

        else:
            raise NotImplementedError(f'Heuristic {heuristic} not implemented for OthelloEnv')

    def __str__(self):
        assert self.parallel_envs == 1
        cur_player_is_o = self.cur_players[0] == 0
        cur_player = 'O' if cur_player_is_o else 'X'
        other_player = 'X' if cur_player_is_o else 'O'
        print('+' + '-+' * (self.config.board_size - 1))
        for i in range(self.config.board_size):
            for j in range(self.config.board_size):
                print('|' + f' {cur_player} ' if self.states[0,0,i,j] == 1 else '|   ', end='')
            print('|')
            print('+' + '-+' * (self.config.board_size - 1))

