from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class EnvConfig:
    pass


class Env:
    def __init__(self, 
        parallel_envs: int,
        config: EnvConfig,
        device: torch.device,
        num_players: int,
        state_shape: torch.Size, 
        policy_shape: torch.Size, 
        value_shape: torch.Size,
        debug: bool = False,
        very_positive_value: float = 1e8
    ):
        self.config = config
        self.parallel_envs = parallel_envs
        self.state_shape = state_shape
        self.policy_shape = policy_shape
        self.value_shape = value_shape
        self.num_players = num_players
        self.debug = debug


        self.states = torch.zeros((self.parallel_envs, *state_shape), dtype=torch.float32, device=device, requires_grad=False)
        self.terminated = torch.zeros(self.parallel_envs, dtype=torch.bool, device=device, requires_grad=False)
        self.rewards = torch.zeros((self.parallel_envs, ), dtype=torch.float32, device=device, requires_grad=False)
        self.cur_players = torch.zeros((self.parallel_envs, ), dtype=torch.long, device=device, requires_grad=False)
        self.env_indices = torch.arange(self.parallel_envs, device=device, requires_grad=False)

        self.device = device
        self.very_positive_value = very_positive_value
        
        
    def reset(self, seed=None):
        raise NotImplementedError()
    
    def step(self, actions) -> torch.Tensor:
        self.push_actions(actions)
        self.next_turn()
        self.update_terminated()
        return self.terminated
    
    def update_terminated(self):
        self.terminated = self.is_terminal()

    def is_terminal(self):
        raise NotImplementedError()
    
    def push_actions(self, actions):
        raise NotImplementedError()
    
    def get_legal_actions(self):
        return torch.ones(self.parallel_envs, *self.policy_shape, dtype=torch.bool, device=self.device, requires_grad=False)

    def apply_stochastic_progressions(self, mask=None) -> None:
        progs, probs = self.get_stochastic_progressions()
        if mask is None:  
            indices = torch.multinomial(probs, 1, replacement=True).flatten()
        else:
            indices = torch.multinomial(probs + (~(mask.unsqueeze(1))), 1, replacement=True).flatten()

        new_states = progs[(self.env_indices, indices)].unsqueeze(1)

        if mask is not None:
            mask = mask.view(self.parallel_envs, 1, 1, 1)
            self.states = self.states * (~mask) + new_states * mask
        else:
            self.states = new_states

    def get_stochastic_progressions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
    
    def reset_terminated_states(self):
        raise NotImplementedError()
    
    def next_player(self):
        self.cur_players = (self.cur_players + 1) % self.num_players
    
    def get_rewards(self):
        return self.rewards
    
    def next_turn(self):
        raise NotImplementedError()
    
    def save_node(self):
        raise NotImplementedError()
    
    def load_node(self, envs):
        raise NotImplementedError()