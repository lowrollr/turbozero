from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class EnvConfig:
    env_type: str


class Env:
    def __init__(self, 
        parallel_envs: int,
        config: EnvConfig,
        device: torch.device,
        num_players: int,
        state_shape: torch.Size, 
        policy_shape: torch.Size, 
        value_shape: torch.Size,
        debug: bool = False
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
        self.cur_players = torch.zeros((self.parallel_envs, ), dtype=torch.long, device=device, requires_grad=False)
        self.env_indices = torch.arange(self.parallel_envs, device=device, requires_grad=False)

        self.device = device
    
    def __str__(self):
        return str(self.states)
        
    def reset(self, seed: Optional[int] = None):
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
    
    def get_nn_input(self):
        return self.states
    
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
    
    def reset_terminated_states(self, seed: Optional[int] = None):
        raise NotImplementedError()
    
    def next_player(self):
        self.cur_players = (self.cur_players + 1) % self.num_players
    
    def get_rewards(self, player_ids: Optional[torch.Tensor] = None):
        raise NotImplementedError()
    
    def next_turn(self):
        raise NotImplementedError()
    
    def save_node(self):
        raise NotImplementedError()
    
    def load_node(self, load_envs, saved):
        raise NotImplementedError()
    
    def get_greedy_rewards(self, player_ids: Optional[torch.Tensor] = None, heuristic: Optional[str] = None):
        # returns instantaneous reward, used in greedy algorithms
        raise NotImplementedError()
    
    def choose_random_legal_action(self) -> torch.Tensor:
        legal_actions = self.get_legal_actions()
        return torch.multinomial(legal_actions.float(), 1, replacement=True).flatten()
    
    def random_rollout(self, num_rollouts: int) -> torch.Tensor:
        saved = self.save_node()
        cumulative_rewards = torch.zeros(self.parallel_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        for _ in range(num_rollouts):
            completed = torch.zeros(self.parallel_envs, dtype=torch.bool, device=self.device, requires_grad=False)
            starting_players = self.cur_players.clone()
            while not completed.all():
                actions = self.choose_random_legal_action()
                terminated = self.step(actions)
                rewards = self.get_rewards(starting_players)
                rewards = ((self.cur_players == starting_players) * rewards) + ((self.cur_players != starting_players) * 1-rewards)
                cumulative_rewards += rewards * terminated * (~completed)
                completed = completed | terminated
            self.load_node(torch.full((self.parallel_envs,), True, dtype=torch.bool, device=self.device), saved)
        cumulative_rewards /= num_rollouts
        return cumulative_rewards
    
    def print_state(self, last_action: Optional[int] = None) -> None:
        pass