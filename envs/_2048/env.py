
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from core.env import Env, EnvConfig
from .torchscripts import get_stochastic_progressions, push_actions, get_legal_actions

@dataclass
class _2048EnvConfig(EnvConfig):
    pass

class _2048Env(Env):
    def __init__(self,
        parallel_envs: int,
        config: _2048EnvConfig, 
        device: torch.device,
        debug=False
    ) -> None:
        super().__init__(
            parallel_envs=parallel_envs,
            config=config,
            device=device,
            num_players=1,
            state_shape=torch.Size([1, 4, 4]),
            policy_shape=torch.Size([4]),
            value_shape=torch.Size([1]),
            debug=debug
        )
                
        if self.debug:
            self.get_stochastic_progressions_ts = get_stochastic_progressions
            self.get_legal_actions_ts = get_legal_actions
            self.push_actions_ts = push_actions
        else:
            self.get_stochastic_progressions_ts = torch.jit.trace(get_stochastic_progressions, ( # type: ignore
                self.states,
            ))

            self.get_legal_actions_ts = torch.jit.trace(get_legal_actions, ( # type: ignore
                self.states,
            ))

            self.push_actions_ts = torch.jit.trace(push_actions, ( # type: ignore
                self.states,
                torch.zeros((self.parallel_envs, ), dtype=torch.int64, device=device)
            ))

        self.saved_states = self.states.clone()

    def reset(self, seed=None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.states.zero_()
        self.terminated.zero_()
        self.apply_stochastic_progressions()
        self.apply_stochastic_progressions()
    
    def reset_terminated_states(self):
        self.states *= torch.logical_not(self.terminated).view(self.parallel_envs, 1, 1, 1)
        self.apply_stochastic_progressions(self.terminated)
        self.apply_stochastic_progressions(self.terminated)
        self.terminated.zero_()

    def next_turn(self):
        self.apply_stochastic_progressions(torch.logical_not(self.terminated))
    
    def get_high_squares(self):
        return torch.amax(self.states, dim=(1, 2, 3))
    
    def get_rewards(self, player_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.rewards
        
    def update_terminated(self) -> None:
        self.terminated = self.is_terminal()
    
    def is_terminal(self):
        return (self.get_legal_actions().sum(dim=1, keepdim=True) == 0).flatten()

    def get_legal_actions(self) -> torch.Tensor:
        return self.get_legal_actions_ts(self.states) # type: ignore

    def get_stochastic_progressions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_stochastic_progressions_ts(self.states) # type: ignore

    def push_actions(self, actions) -> None:
        self.states = self.push_actions_ts(self.states, actions) # type: ignore

    def save_node(self) -> torch.Tensor:
        return self.states.clone()
    
    def load_node(self, load_envs: torch.Tensor, saved: torch.Tensor):
        load_envs_expnd = load_envs.view(self.parallel_envs, 1, 1, 1)
        self.states = saved.clone() * load_envs_expnd + self.states * (~load_envs_expnd)
        self.update_terminated()
    
    def __str__(self) -> str:
        envstr = []
        assert self.parallel_envs == 1
        envstr.append('+' + '--------+' * 4)
        envstr.append('\n')
        for i in range(4):
            envstr.append('|' + '        |' * 4)
            envstr.append('\n')
            for j in range(4):
                if self.states[0, 0, i, j] == 0:
                    envstr.append('|        ')
                else:
                    envstr.append(f'|{str(int(2**self.states[0, 0, i, j])).center(8)}')
            envstr.append('|')
            envstr.append('\n')
            envstr.append('|' + '        |' * 4)
            envstr.append('\n')
            envstr.append('+' + '--------+' * 4)
            envstr.append('\n')
        return ''.join(envstr)