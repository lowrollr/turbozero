
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from core.env import Env, EnvConfig

@dataclass
class ConnectXConfig(EnvConfig):
    board_width: int
    board_height: int
    inarow: int

class ConnectXEnv(Env):
    def __init__(self,
        parallel_envs: int,
        config: ConnectXConfig,
        device: torch.device,
        debug: bool = False 
    ):
        self.config = config
        self.parallel_envs = parallel_envs
        state_shape = torch.Size([2, config.board_height, config.board_width])
        policy_shape = torch.Size([config.board_width])
        value_shape = torch.Size([1])

        super().__init__(
            parallel_envs=parallel_envs,
            config=config,
            device=device,
            num_players=2,
            state_shape=state_shape,
            policy_shape=policy_shape,
            value_shape=value_shape,
            debug=debug
        )   

        self.next_empty = torch.full(
            (self.parallel_envs, self.config.board_width),
            self.config.board_height - 1,
            dtype=torch.int64,
            device=self.device,
            requires_grad=False
        )
        self.reward_check = torch.zeros(
            (self.parallel_envs, 2),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        self.reward_check[:, 0] = self.config.inarow
        self.reward_check[:, 1] = -self.config.inarow

        self.kernels = self.create_kernels()

    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = 0
        self.states.zero_()
        self.terminated.zero_()
        self.cur_players.zero_()
        self.next_empty.fill_(self.config.board_height - 1)
        return seed

    def push_actions(self, actions):
        self.states[self.env_indices, 0, self.next_empty[self.env_indices, actions], actions] = 1
        self.next_empty[self.env_indices, actions] -= 1

    def create_kernels(self):
        size = self.config.inarow

        horiz = torch.zeros((2, size, size), dtype=torch.float32, device=self.device, requires_grad=False)
        horiz[0, 0, :] = 1
        horiz[1, 0, :] = -1
        vert = torch.zeros((2, size, size), dtype=torch.float32, device=self.device, requires_grad=False)
        vert[1, :, 0] = 1
        vert[1, :, 0] = -1
        diag = torch.eye(size, device=self.device)
        inv_diag = torch.flip(torch.eye(size, device=self.device), dims=(0,))

        kernels = torch.stack(
            [
                torch.stack((diag, -diag), dim=0),
                torch.stack((inv_diag, -inv_diag), dim=0),
                horiz,
                vert
            ]
        )
        return kernels

    def get_legal_actions(self) -> torch.Tensor:
        return self.next_empty >= 0
    
    def next_turn(self, *args, **kwargs):
        self.states = torch.roll(self.states, 1, dims=1)
        self.next_player()

    def save_node(self):
        return (
            self.states.clone(),
            self.cur_players.clone(),
            self.next_empty.clone()
        )
    
    def load_node(self, load_envs: torch.Tensor, saved: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        load_envs_expnd = load_envs.view(-1, 1, 1, 1)
        self.states = saved[0].clone() * load_envs_expnd + self.states * (~load_envs_expnd)
        self.cur_players = saved[1].clone() * load_envs + self.cur_players * (~load_envs)
        self.next_empty = saved[2].clone() * load_envs.view(-1, 1) + self.next_empty * (~load_envs).view(-1, 1)
        self.update_terminated()
    
    def reset_terminated_states(self, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = 0
        terminated = self.terminated.clone()
        self.states *= 1 * ~terminated.view(-1, 1, 1, 1)
        self.cur_players *= 1 * ~terminated
        self.next_empty *= 1 * ~terminated.view(-1, 1)
        self.next_empty += (self.config.board_height - 1) * terminated.view(-1, 1)
        self.terminated.zero_()
        return seed

    
    def is_terminal(self):
        return (self.get_rewards() != 0.5) | ~(self.get_legal_actions().any(dim=1))

    def get_rewards(self, player_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if player_ids is None:
            player_ids = self.cur_players
        idx = ((player_ids == self.cur_players).int() - 1) % 2
        other_idx = 1 - idx
        convolved = torch.functional.F.conv2d(self.states, self.kernels, padding=(self.config.inarow - 1, self.config.inarow - 1)).view(self.parallel_envs, -1)
        p1_rewards = (convolved == self.reward_check[self.env_indices, idx].view(self.parallel_envs, 1)).any(dim=1)
        p2_rewards = (convolved == self.reward_check[self.env_indices, other_idx].view(self.parallel_envs, 1)).any(dim=1)
        rewards = (1 * (p1_rewards > p2_rewards)) + (0.5 * (p1_rewards == p2_rewards))
        return rewards
    
    



