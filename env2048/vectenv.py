
from typing import Optional, Tuple
import torch

class Vectorized2048Env:
    def __init__(self, num_parallel_envs, device, progression_batch_size):
        self.num_parallel_envs = num_parallel_envs
        self.boards = torch.zeros((num_parallel_envs, 1, 4, 4), dtype=torch.float32, device=device, requires_grad=False)
        self.invalid_mask = torch.zeros((num_parallel_envs, 1, 1, 1), dtype=torch.bool, device=device, requires_grad=False)
        self.device = device
        self.very_negative_value = -1e5

        ones = torch.eye(16, dtype=torch.float32, requires_grad=False).view(16, 4, 4)
        twos = torch.eye(16, dtype=torch.float32, requires_grad=False).view(16, 4, 4) * 2
        self.base_progressions = torch.concat([ones, twos], dim=0).to(device)
        self.base_probabilities = torch.concat([torch.full((16,), 0.9, requires_grad=False), torch.full((16,), 0.1, requires_grad=False)], dim=0).to(device)

        self.mask0 = torch.tensor([[[[self.very_negative_value, 1]]]], dtype=torch.float32, device=device, requires_grad=False)
        self.mask1 = torch.tensor([[[[1], [self.very_negative_value]]]], dtype=torch.float32, device=device, requires_grad=False)
        self.mask2 = torch.tensor([[[[1, self.very_negative_value]]]], dtype=torch.float32, device=device, requires_grad=False)
        self.mask3 = torch.tensor([[[[self.very_negative_value], [1]]]], dtype=torch.float32, device=device, requires_grad=False)
        
        self.env_indices = torch.arange(num_parallel_envs, device=device, requires_grad=False)

        self.progression_batch_size = progression_batch_size

        self.fws_cont = torch.ones(num_parallel_envs, dtype=torch.bool, device=device, requires_grad=False)
        self.fws_sums = torch.zeros(num_parallel_envs, dtype=torch.float32, device=device, requires_grad=False)
        self.fws_res = torch.zeros(num_parallel_envs, dtype=torch.int64, device=device, requires_grad=False)
        self.randn = torch.zeros(num_parallel_envs, dtype=torch.float32, device=device, requires_grad=False)

        self.fws_cont_batch = torch.ones(progression_batch_size, dtype=torch.bool, device=device, requires_grad=False)
        self.fws_sums_batch = torch.zeros(progression_batch_size, dtype=torch.float32, device=device, requires_grad=False)
        self.fws_res_batch = torch.zeros(progression_batch_size, dtype=torch.int64, device=device, requires_grad=False)
        self.randn_batch = torch.zeros(progression_batch_size, dtype=torch.float32, device=device, requires_grad=False)

        self.rank = torch.arange(4, 0, -1, device=device).expand((self.num_parallel_envs * 4, 4))

    def fast_weighted_sample(self, weights, norm=True, generator=None): # yields > 50% speedup over torch.multinomial for our use-cases!
        # weights.div_(weights.sum(dim=1, keepdim=True))
        if norm:
            nweights = weights.div(weights.sum(dim=1, keepdim=True))
        else:
            nweights = weights

        num_samples = nweights.shape[0]
        num_categories = nweights.shape[1]

        if num_samples == self.progression_batch_size:
            self.fws_cont_batch.fill_(1)
            self.fws_sums_batch.zero_()
            self.fws_res_batch.zero_()
            self.randn_batch.uniform_(0, 1, generator = generator)
            conts, sums, res, rand_vals = self.fws_cont_batch, self.fws_sums_batch, self.fws_res_batch, self.randn_batch
        else:
            self.fws_cont.fill_(1)
            self.fws_sums.zero_()
            self.fws_res.zero_()
            self.randn.uniform_(0, 1, generator = generator)
            conts, sums, res, rand_vals = self.fws_cont, self.fws_sums, self.fws_res, self.randn

        for i in range(num_categories - 1):
            sums.add_(nweights[:, i])
            cont = rand_vals.gt(sums)
            res.add_(torch.logical_not(cont) * i * conts)
            conts.mul_(cont)
        res.add_(conts * (num_categories - 1))
        
        return res
    

    def reset(self, seed=None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.boards.fill_(0)
        self.invalid_mask.fill_(0)
        self.spawn_tiles()
        self.spawn_tiles()
        
    def step(self, actions: torch.Tensor) -> torch.Tensor:
        self.push_moves(actions)
        self.spawn_tiles(torch.logical_not(self.invalid_mask))
        self.update_invalid_mask()
        return self.invalid_mask
    
    def reset_invalid_boards(self):
        self.boards *= torch.logical_not(self.invalid_mask)
        self.spawn_tiles(self.invalid_mask)
        self.spawn_tiles(self.invalid_mask)
        self.invalid_mask.fill_(0)
    
    def get_high_squares(self):
        return torch.amax(self.boards, dim=(1, 2, 3))
        
    def update_invalid_mask(self) -> None:
        self.invalid_mask = (self.get_legal_moves().sum(dim=1, keepdim=True) == 0).view(-1, 1, 1, 1)

    def get_legal_moves(self) -> torch.Tensor:
        # check for empty spaces        
        m0 = torch.nn.functional.conv2d(self.boards, self.mask0, padding=0, bias=None).view(self.num_parallel_envs, 12)
        m1 = torch.nn.functional.conv2d(self.boards, self.mask1, padding=0, bias=None).view(self.num_parallel_envs, 12)
        m2 = torch.nn.functional.conv2d(self.boards, self.mask2, padding=0, bias=None).view(self.num_parallel_envs, 12)
        m3 = torch.nn.functional.conv2d(self.boards, self.mask3, padding=0, bias=None).view(self.num_parallel_envs, 12)

        m0_valid = torch.any(m0 > 0.5, dim=1, keepdim=True)
        m1_valid = torch.any(m1 > 0.5, dim=1, keepdim=True)
        m2_valid = torch.any(m2 > 0.5, dim=1, keepdim=True)
        m3_valid = torch.any(m3 > 0.5, dim=1, keepdim=True)

        # check for matching tiles
        vertical_comparison = torch.any((torch.logical_and(self.boards[:,:,:-1,:] == self.boards[:,:,1:,:], self.boards[:,:,1:,:] != 0)).view(self.num_parallel_envs, 12), dim=1, keepdim=True)
        horizontal_comparison = torch.any((torch.logical_and(self.boards[:,:,:,:-1] == self.boards[:,:,:,1:], self.boards[:,:,:,1:] != 0)).view(self.num_parallel_envs, 12), dim=1, keepdim=True)

        m0_valid.logical_or_(horizontal_comparison)
        m2_valid.logical_or_(horizontal_comparison)
        m1_valid.logical_or_(vertical_comparison)
        m3_valid.logical_or_(vertical_comparison)

        return torch.concat([m0_valid, m1_valid, m2_valid, m3_valid], dim=1)

    def get_progressions(self, boards_batch) -> Tuple[torch.Tensor, torch.Tensor]:
        # check and see if each of the progressions are valid (no tile already in that spot)
        # base_progressions is a 32x4x4 tensor with all the possible progressions
        # bs is an Nx4x4 tensor with N board states
        # returns an 32xNx4x4 tensor with 32 possible progressions for each board state
        valid_progressions = torch.logical_not(torch.any((boards_batch * self.base_progressions).view(-1, 32, 16), dim=2)).view(-1, 32, 1, 1)
        progressions = (boards_batch + self.base_progressions) * valid_progressions
        probs = self.base_probabilities * valid_progressions.view(-1, 32)
        return progressions, probs

    def spawn_tiles(self, mask=None) -> None:
        start_index = 0
        while start_index < self.num_parallel_envs:
            end_index = min(start_index + self.progression_batch_size, self.num_parallel_envs)
            boards_batch = self.boards[start_index:end_index]
            progs, probs = self.get_progressions(boards_batch)
            indices = self.fast_weighted_sample(probs, norm=True).unsqueeze(1)
            if mask is not None:
                self.boards[start_index:end_index] = torch.where(mask[start_index:end_index], progs[(self.env_indices[:end_index-start_index], indices[:,0])].unsqueeze(1), boards_batch)
            else:
                self.boards[start_index:end_index] = progs[(self.env_indices[:end_index-start_index], indices[:,0])].unsqueeze(1)

            start_index = end_index
    
    def rotate_by_amnts(self, amnts) -> None:
        amnts = amnts.view(-1, 1, 1, 1).expand_as(self.boards)    
        self.boards = torch.where(amnts == 1, self.boards.flip(2).transpose(2,3), self.boards)
        self.boards = torch.where(amnts == 2, self.boards.flip(3).flip(2), self.boards)
        self.boards = torch.where(amnts == 3, self.boards.flip(3).transpose(2,3), self.boards)

 
    def customized_merge_sort(self, bs_flat):
        non_zero_mask = (bs_flat != 0)

        # Set the rank of zero elements to zero
        rank = self.rank * non_zero_mask

        # Create a tensor of sorted indices by sorting the rank tensor along dim=-1
        sorted_indices = torch.argsort(rank, dim=-1, descending=True)

        sorted_bs_flat = torch.zeros_like(bs_flat)
        sorted_bs_flat.scatter_(1, sorted_indices, bs_flat)
        return sorted_bs_flat
    
    def merge(self) -> None:
        shape = self.boards.shape
        bs_flat = self.boards.view(-1, shape[-1])

        # Step 1: initial sort using a customized version of merge sort
        bs_flat = self.customized_merge_sort(bs_flat)

        # Step 2: apply merge operation
        for i in range(3):
            is_same = torch.logical_and(bs_flat[:,i] == bs_flat[:,i+1], bs_flat[:,i] != 0)
            bs_flat[:,i].add_(is_same)
            bs_flat[:,i+1].masked_fill_(is_same, 0)

        # Step 3: reapply the customized merge sort
        bs_flat = self.customized_merge_sort(bs_flat)

        self.boards = bs_flat.view(shape)

    def push_moves(self, moves) -> None:
        self.rotate_by_amnts(moves)
        self.merge()
        self.rotate_by_amnts((4-moves) % 4)