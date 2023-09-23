
from dataclasses import dataclass
from typing import Optional, Tuple
from colorama import Fore
import torch
from core.env import Env, EnvConfig
from .torchscripts import get_stochastic_progressions, push_actions, get_legal_actions

COLOR_MAP = {
    1: Fore.WHITE,
    2: Fore.LIGHTWHITE_EX,
    3: Fore.LIGHTYELLOW_EX,
    4: Fore.LIGHTRED_EX,
    5: Fore.LIGHTMAGENTA_EX,
    6: Fore.LIGHTGREEN_EX,
    7: Fore.LIGHTCYAN_EX,
    8: Fore.LIGHTBLUE_EX,
    9: Fore.YELLOW,
    10: Fore.RED,
    11: Fore.MAGENTA,
    12: Fore.GREEN,
    13: Fore.CYAN,
    14: Fore.BLUE,
    15: Fore.WHITE,
    16: Fore.BLACK,
    17: Fore.LIGHTBLACK_EX
}


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
        # self.rewards is just a dummy tensor here
        self.rewards = torch.zeros((self.parallel_envs, ), dtype=torch.float32, device=device, requires_grad=False)

    def reset(self, seed=None) -> int:
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = 0
        self.states.zero_()
        self.terminated.zero_()
        self.apply_stochastic_progressions()
        self.apply_stochastic_progressions()
        return seed
    
    def reset_terminated_states(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            torch.manual_seed()
        else:
            seed = 0
        self.states *= torch.logical_not(self.terminated).view(self.parallel_envs, 1, 1, 1)
        self.apply_stochastic_progressions(self.terminated)
        self.apply_stochastic_progressions(self.terminated)
        self.terminated.zero_()
        return seed

    def next_turn(self):
        self.apply_stochastic_progressions(torch.logical_not(self.terminated))
    
    def get_high_squares(self):
        return torch.amax(self.states, dim=(1, 2, 3))
    
    def get_rewards(self, player_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TODO: handle rewards in env instead of collector postprocessing
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
    
    def print_state(self, action=None) -> None:
        envstr = []
        assert self.parallel_envs == 1
        envstr.append((Fore.BLUE if action == 3 else '') + '+' + '--------+' * 4)

        envstr.append(Fore.RESET + '\n')
        for i in range(4):
            envstr.append((Fore.BLUE if action == 0 else ''))
            envstr.append('|' + Fore.RESET + '        |' * 3)
            envstr.append((Fore.BLUE if action == 2 else ''))
            envstr.append('        |')
            envstr.append(Fore.RESET + '\n')
            for j in range(4):
                color = Fore.RESET
                if j == 0 and action == 0:
                    color = Fore.BLUE
                if self.states[0, 0, i, j] == 0:
                    envstr.append(color + '|        ')
                else:
                    num = int(self.states[0, 0, i, j])
                    envstr.append(color + '|' + COLOR_MAP[num] + str(2**num).center(8))
            envstr.append((Fore.BLUE if action == 2 else Fore.RESET))
            envstr.append('|')
            envstr.append(Fore.RESET + '\n')
            envstr.append((Fore.BLUE if action == 0 else ''))
            envstr.append('|' + Fore.RESET + '        |' * 3)
            envstr.append((Fore.BLUE if action == 2 else ''))
            envstr.append('        |')
            envstr.append(Fore.RESET + '\n')
            if i < 3:
                envstr.append((Fore.BLUE if action == 0 else ''))
                envstr.append('+' + Fore.RESET + '--------+' * 3)
                envstr.append('--------' +(Fore.BLUE if action == 2 else '') + '+')
            else:
                envstr.append((Fore.BLUE if action == 1 else ''))
                envstr.append('+' + '--------+' * 4)
            envstr.append(Fore.RESET + '\n')
        print(''.join(envstr))