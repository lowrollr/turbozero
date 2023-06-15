
from typing import Optional, Tuple
import torch
from core import GLOB_FLOAT_TYPE
from core.vectenv import VectEnv
from .torchscripts import get_stochastic_progressions, push_actions, get_legal_actions

class _2048Env(VectEnv):
    def __init__(self, num_parallel_envs, device):
        super().__init__(
            num_parallel_envs=num_parallel_envs,
            state_shape=torch.Size([1, 4, 4]),
            policy_shape=torch.Size([4]),
            value_shape=torch.Size([1]),
            device=device, 
            is_stochastic=True
        )
        # self.mask0 = torch.tensor([[[[self.very_negative_value, 1]]]], dtype=GLOB_FLOAT_TYPE, device=device, requires_grad=False)
        # self.mask1 = torch.tensor([[[[1], [self.very_negative_value]]]], dtype=GLOB_FLOAT_TYPE, device=device, requires_grad=False)
        # self.mask2 = torch.tensor([[[[1, self.very_negative_value]]]], dtype=GLOB_FLOAT_TYPE, device=device, requires_grad=False)
        # self.mask3 = torch.tensor([[[[self.very_negative_value], [1]]]], dtype=GLOB_FLOAT_TYPE, device=device, requires_grad=False)

        self.push_actions_ts = torch.jit.trace(push_actions, (
            self.states,
            torch.zeros((self.num_parallel_envs, ), dtype=torch.int64, device=device)
        ))

        self.get_legal_actions_ts = torch.jit.trace(get_legal_actions, (
            self.states,
        ))

        self.get_stochastic_progressions_ts = torch.jit.trace(get_stochastic_progressions, (
            self.states,
        ))

    def reset(self, seed=None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.states.zero_()
        self.invalid_mask.zero_()
        self.stochastic_step()
        self.stochastic_step()
    
    def reset_invalid_states(self):
        self.states *= torch.logical_not(self.invalid_mask).view(self.num_parallel_envs, 1, 1, 1)
        self.stochastic_step(self.invalid_mask)
        self.stochastic_step(self.invalid_mask)
        self.invalid_mask.zero_()
    
    def get_high_squares(self):
        return torch.amax(self.states, dim=(1, 2, 3))
        
    def update_invalid_mask(self) -> None:
        self.invalid_mask = (self.get_legal_actions().sum(dim=1, keepdim=True) == 0).flatten()

    def get_legal_actions(self) -> torch.Tensor:
        return self.get_legal_actions_ts(self.states)

    def get_stochastic_progressions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_stochastic_progressions_ts(self.states)

    def push_actions(self, actions) -> None:
        self.states = self.push_actions_ts(self.states, actions)