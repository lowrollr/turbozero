
from typing import Optional, Tuple
import torch

class Vectorized2048Env:
    def __init__(self, num_parallel_envs, device):
        self.num_parallel_envs = num_parallel_envs
        self.boards = torch.zeros((num_parallel_envs, 4, 4), dtype=torch.int32, device=device)
        self.reset_mask = torch.zeros(num_parallel_envs, dtype=torch.bool, device=device)
        self.mcts_trees = [None for _ in range(num_parallel_envs)]
        self.device = device
        self.very_negative_value = -1e5

        ones = torch.eye(16, dtype=torch.int16).view(16, 4, 4)
        twos = torch.eye(16, dtype=torch.int16).view(16, 4, 4) * 2
        self.base_progressions = torch.concat([ones, twos], dim=0)
        self.base_probabilities = torch.concat([torch.full((16,), 0.9), torch.full((16,), 0.1)], dim=0)

    def reset(self, seed=None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        
    def step(self, actions: torch.Tensor):
        self.push_moves(actions)
        self.reset_mask = self.check_terminal()
        self.spawn_tiles()
        

    def check_terminal(self) -> torch.Tensor:
        return self.get_legal_moves().sum(dim=1) == 0

    def get_legal_moves(self) -> torch.Tensor:
        # check for empty spaces
        mask0 = torch.tensor([[[[self.very_negative_value, 1]]]], dtype=torch.int16)
        mask1 = torch.tensor([[[[1, self.very_negative_value]]]], dtype=torch.int16)
        mask2 = torch.tensor([[[[1], [self.very_negative_value]]]], dtype=torch.int16)
        mask3 = torch.tensor([[[[self.very_negative_value], [1]]]], dtype=torch.int16)
        
        m0 = torch.nn.functional.conv2d(self.boards, mask0, padding=0, bias=None).view(self.num_parallel_envs, 12)
        m1 = torch.nn.functional.conv2d(self.boards, mask1, padding=0, bias=None).view(self.num_parallel_envs, 12)
        m2 = torch.nn.functional.conv2d(self.boards, mask2, padding=0, bias=None).view(self.num_parallel_envs, 12)
        m3 = torch.nn.functional.conv2d(self.boards, mask3, padding=0, bias=None).view(self.num_parallel_envs, 12)

        m0_valid = torch.any(m0 > 0, dim=1, keepdim=True)
        m1_valid = torch.any(m1 > 0, dim=1, keepdim=True)
        m2_valid = torch.any(m2 > 0, dim=1, keepdim=True)
        m3_valid = torch.any(m3 > 0, dim=1, keepdim=True)

        # check for matching tiles
        vertical_comparison = torch.any((torch.logical_and(self.boards[:,:,:-1,:] == self.boards[:,:,1:,:], self.boards[:,:,1:,:] != 0)).view(self.num_parallel_envs, 12), dim=1, keepdim=True)
        horizontal_comparison = torch.any((torch.logical_and(self.boards[:,:,:,:-1] == self.boards[:,:,:,1:], self.boards[:,:,:,1:] != 0)).view(self.num_parallel_envs, 12), dim=1, keepdim=True)

        m0_valid = torch.logical_or(m0_valid, horizontal_comparison)
        m1_valid = torch.logical_or(m1_valid, horizontal_comparison)

        m2_valid = torch.logical_or(m2_valid, vertical_comparison)
        m3_valid = torch.logical_or(m3_valid, vertical_comparison)

        return torch.concat([m0_valid, m1_valid, m2_valid, m3_valid], dim=1)

    def get_progressions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # check and see if each of the progressions are valid (no tile already in that spot)
        # base_progressions is a 32x4x4 tensor with all the possible progressions
        # bs is an Nx4x4 tensor with N board states
        # returns an 32xNx4x4 tensor with 32 possible progressions for each board state
        valid_progressions = torch.logical_not(torch.any((self.boards * self.base_progressions).view(self.num_parallel_envs, 32, 16), dim=2)).view(self.num_parallel_envs, 32, 1, 1)
        progressions = (self.boards + self.base_progressions) * valid_progressions
        probs = self.base_probabilities * valid_progressions.view(self.num_parallel_envs, 32)
        return progressions, probs

    def spawn_tiles(self) -> None:
        progs, probs = self.get_progressions()
        probs += torch.where(probs.amax(dim=1, keepdim=True) == 0, torch.full_like(probs, 1), torch.full_like(probs, 0))
        indices = torch.multinomial(probs, 1)
        self.boards = progs[(range(self.num_parallel_envs), indices[:,0])]
    
    def rotate_by_amnts(self, amnts) -> None:
        rotations_0 = self.boards
        rotations_1 = torch.rot90(self.boards, 1, (2,3))
        rotations_2 = torch.rot90(self.boards, 2, (2,3))
        rotations_3 = torch.rot90(self.boards, 3, (2,3))
        mask_0 = (amnts == 0)
        mask_90 = (amnts == 1)
        mask_180 = (amnts == 2)
        mask_270 = (amnts == 3)    
        self.boards[mask_0] = rotations_0[mask_0]
        self.boards[mask_90] = rotations_1[mask_90]
        self.boards[mask_180] = rotations_2[mask_180]
        self.boards[mask_270] = rotations_3[mask_270]

    def merge(self) -> None:
        shape = self.boards.shape
        bs_flat = self.boards.view(-1, shape[-1])
        mask = (bs_flat != 0)
        _, sorted_indices = torch.sort(mask, dim=1, descending=True)
        bs_flat = torch.gather(bs_flat, 1, sorted_indices)
        for i in range(3):
            is_same = torch.logical_and(bs_flat[:,i] == bs_flat[:,i+1], bs_flat[:,i] != 0).int()
            bs_flat[:,i] += is_same
            bs_flat[:,i+1] *= (1 - is_same)
        mask = (bs_flat != 0)
        _, sorted_indices = torch.sort(mask, dim=1, descending=True)
        bs_flat = torch.gather(bs_flat, 1, sorted_indices)
        self.boards = bs_flat.view(shape)

    def push_moves(self, moves) -> None:
        self.rotate_by_amnts(moves)
        self.merge()
        self.rotate_by_amnts((4-moves) % 4)
    
