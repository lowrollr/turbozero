
from typing import Optional, Tuple
import torch
from core import GLOB_FLOAT_TYPE
from core.vectenv import VectEnv

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

        self.very_negative_value = -1e5

        ones = torch.eye(16, dtype=GLOB_FLOAT_TYPE, requires_grad=False).view(16, 4, 4)
        twos = torch.eye(16, dtype=GLOB_FLOAT_TYPE, requires_grad=False).view(16, 4, 4) * 2
        self.base_progressions = torch.concat([ones, twos], dim=0).to(device)
        self.base_probabilities = torch.concat([torch.full((16,), 0.9, requires_grad=False), torch.full((16,), 0.1, requires_grad=False)], dim=0).to(device)

        self.mask0 = torch.tensor([[[[self.very_negative_value, 1]]]], dtype=GLOB_FLOAT_TYPE, device=device, requires_grad=False)
        self.mask1 = torch.tensor([[[[1], [self.very_negative_value]]]], dtype=GLOB_FLOAT_TYPE, device=device, requires_grad=False)
        self.mask2 = torch.tensor([[[[1, self.very_negative_value]]]], dtype=GLOB_FLOAT_TYPE, device=device, requires_grad=False)
        self.mask3 = torch.tensor([[[[self.very_negative_value], [1]]]], dtype=GLOB_FLOAT_TYPE, device=device, requires_grad=False)
    
        self.rank = torch.arange(4, 0, -1, device=device, requires_grad=False).expand((self.num_parallel_envs * 4, 4))

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
        # check for empty spaces        
        m0 = torch.nn.functional.conv2d(self.states, self.mask0, padding=0, bias=None).view(self.num_parallel_envs, 12)
        m1 = torch.nn.functional.conv2d(self.states, self.mask1, padding=0, bias=None).view(self.num_parallel_envs, 12)
        m2 = torch.nn.functional.conv2d(self.states, self.mask2, padding=0, bias=None).view(self.num_parallel_envs, 12)
        m3 = torch.nn.functional.conv2d(self.states, self.mask3, padding=0, bias=None).view(self.num_parallel_envs, 12)

        m0_valid = torch.any(m0 > 0.5, dim=1, keepdim=True)
        m1_valid = torch.any(m1 > 0.5, dim=1, keepdim=True)
        m2_valid = torch.any(m2 > 0.5, dim=1, keepdim=True)
        m3_valid = torch.any(m3 > 0.5, dim=1, keepdim=True)

        # check for matching tiles
        vertical_comparison = torch.any((torch.logical_and(self.states[:,:,:-1,:] == self.states[:,:,1:,:], self.states[:,:,1:,:] != 0)).view(self.num_parallel_envs, 12), dim=1, keepdim=True)
        horizontal_comparison = torch.any((torch.logical_and(self.states[:,:,:,:-1] == self.states[:,:,:,1:], self.states[:,:,:,1:] != 0)).view(self.num_parallel_envs, 12), dim=1, keepdim=True)

        m0_valid.logical_or_(horizontal_comparison)
        m2_valid.logical_or_(horizontal_comparison)
        m1_valid.logical_or_(vertical_comparison)
        m3_valid.logical_or_(vertical_comparison)

        return torch.concat([m0_valid, m1_valid, m2_valid, m3_valid], dim=1)

    def get_stochastic_progressions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # check and see if each of the progressions are valid (no tile already in that spot)
        # base_progressions is a 32x4x4 tensor with all the possible progressions
        # bs is an Nx4x4 tensor with N board states
        # returns an 32xNx4x4 tensor with 32 possible progressions for each board state
        valid_progressions = torch.logical_not(torch.any((self.states * self.base_progressions).view(-1, 32, 16), dim=2))
        progressions = (self.states + self.base_progressions) * valid_progressions.view(self.num_parallel_envs, 32, 1, 1)
        probs = self.base_probabilities * valid_progressions
        return progressions, probs
    
    def rotate_by_amnts(self, amnts) -> None:
        amnts = amnts.view(-1, 1, 1, 1).expand_as(self.states)    
        self.states = torch.where(amnts == 1, self.states.flip(2).transpose(2,3), self.states)
        self.states = torch.where(amnts == 2, self.states.flip(3).flip(2), self.states)
        self.states = torch.where(amnts == 3, self.states.flip(3).transpose(2,3), self.states)

    def collapse(self, bs_flat):
        non_zero_mask = (bs_flat != 0)

        # Set the rank of zero elements to zero
        rank = (self.rank * non_zero_mask)

        # Create a tensor of sorted indices by sorting the rank tensor along dim=-1
        sorted_indices = torch.argsort(rank, dim=-1, descending=True, stable=True)

        return torch.gather(bs_flat, dim=-1, index=sorted_indices)
    
    def merge(self) -> None:
        shape = self.states.shape
        bs_flat = self.states.view(-1, shape[-1])

        # Step 1: initial sort using a customized version of merge sort
        bs_flat = self.collapse(bs_flat)

        # Step 2: apply merge operation
        for i in range(3):
            is_same = torch.logical_and(bs_flat[:,i] == bs_flat[:,i+1], bs_flat[:,i] != 0)
            bs_flat[:,i].add_(is_same)
            bs_flat[:,i+1].masked_fill_(is_same, 0)

        # Step 3: reapply the customized merge sort
        bs_flat = self.collapse(bs_flat)

        self.states = bs_flat.view(shape)

    def push_actions(self, actions) -> None:
        self.rotate_by_amnts(actions)
        self.merge()
        self.rotate_by_amnts((4-actions) % 4)