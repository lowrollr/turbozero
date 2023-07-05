
from typing import Tuple
import torch

def collapse(rankt, bs_flat) -> torch.Tensor: 
    non_zero_mask = (bs_flat != 0)

    # Set the rank of zero elements to zero
    rank = (rankt * non_zero_mask)

    # Create a tensor of sorted indices by sorting the rank tensor along dim=-1
    sorted_indices = torch.argsort(rank, dim=-1, descending=True, stable=True)

    return torch.gather(bs_flat, dim=-1, index=sorted_indices)


def merge(states, rankt) -> torch.Tensor:
    shape = states.shape
    bs_flat = states.view(-1, shape[-1])

    # Step 1: initial sort using a customized version of merge sort
    bs_flat = collapse(rankt, bs_flat)

    # Step 2: apply merge operation
    for i in range(3):
        is_same = torch.logical_and(bs_flat[:,i] == bs_flat[:,i+1], bs_flat[:,i] != 0)
        bs_flat[:,i].add_(is_same)
        bs_flat[:,i+1].masked_fill_(is_same, 0)

    # Step 3: reapply the customized merge sort
    bs_flat = collapse(rankt, bs_flat)

    return bs_flat.view(shape)

def rotate_by_amnts(states, amnts):
    mask1 = amnts == 1
    mask2 = amnts == 2
    mask3 = amnts == 3

    states[mask1] = states.flip(2).transpose(2, 3)[mask1]
    states[mask2] = states.flip(3).flip(2)[mask2]
    states[mask3] = states.flip(3).transpose(2, 3)[mask3]

    return states

def push_actions(states, actions) -> torch.Tensor:
    rankt = torch.arange(4, 0, -1, device=states.device, requires_grad=False).expand((states.shape[0] * 4, 4))
    actions = actions.view(-1, 1, 1, 1).expand_as(states)   
    states = rotate_by_amnts(states, actions)
    states = merge(states, rankt)
    states = rotate_by_amnts(states, (4-actions) % 4)
    return states

def get_legal_actions(states):
    n = states.shape[0]
    device = states.device
    dtype = states.dtype

    mask0 = torch.tensor([[[[-1e5, 1]]]], dtype=dtype, device=device, requires_grad=False)
    mask1 = torch.tensor([[[[1], [-1e5]]]], dtype=dtype, device=device, requires_grad=False)
    mask2 = torch.tensor([[[[1, -1e5]]]], dtype=dtype, device=device, requires_grad=False)
    mask3 = torch.tensor([[[[-1e5], [1]]]], dtype=dtype, device=device, requires_grad=False)

    m0 = torch.nn.functional.conv2d(states, mask0, padding=0, bias=None).view(n, 12)
    m1 = torch.nn.functional.conv2d(states, mask1, padding=0, bias=None).view(n, 12)
    m2 = torch.nn.functional.conv2d(states, mask2, padding=0, bias=None).view(n, 12)
    m3 = torch.nn.functional.conv2d(states, mask3, padding=0, bias=None).view(n, 12)

    m0_valid = torch.any(m0 > 0.5, dim=1, keepdim=True)
    m1_valid = torch.any(m1 > 0.5, dim=1, keepdim=True)
    m2_valid = torch.any(m2 > 0.5, dim=1, keepdim=True)
    m3_valid = torch.any(m3 > 0.5, dim=1, keepdim=True)

    # Compute the differences between adjacent elements in the 2nd and 3rd dimensions
    vertical_diff = states[:, :, :-1, :] - states[:, :, 1:, :]
    horizontal_diff = states[:, :, :, :-1] - states[:, :, :, 1:]

    # Check where the differences are zero, excluding the zero elements in the original matrix
    vertical_zeros = torch.logical_and(vertical_diff == 0, states[:, :, 1:, :] != 0)
    horizontal_zeros = torch.logical_and(horizontal_diff == 0, states[:, :, :, 1:] != 0)

    # Flatten the last two dimensions and compute the logical OR along the last dimension
    vertical_comparison = vertical_zeros.view(n, 12).any(dim=1, keepdim=True)
    horizontal_comparison = horizontal_zeros.view(n, 12).any(dim=1, keepdim=True)
    m0_valid.logical_or_(horizontal_comparison)
    m2_valid.logical_or_(horizontal_comparison)
    m1_valid.logical_or_(vertical_comparison)
    m3_valid.logical_or_(vertical_comparison)

    return torch.concat([m0_valid, m1_valid, m2_valid, m3_valid], dim=1)

def get_stochastic_progressions(states) -> Tuple[torch.Tensor, torch.Tensor]:
    ones = torch.eye(16, dtype=states.dtype).view(16, 4, 4)
    twos = torch.eye(16, dtype=states.dtype).view(16, 4, 4) * 2
    base_progressions = torch.concat([ones, twos], dim=0).to(states.device)
    base_probabilities = torch.concat([torch.full((16,), 0.9), torch.full((16,), 0.1)], dim=0).to(states.device)
    # check and see if each of the progressions are valid (no tile already in that spot)
    # base_progressions is a 32x4x4 tensor with all the possible progressions
    # bs is an Nx4x4 tensor with N board states
    # returns an 32xNx4x4 tensor with 32 possible progressions for each board state
    valid_progressions = torch.logical_not(torch.any((states * base_progressions).view(-1, 32, 16), dim=2))
    progressions = (states + base_progressions) * valid_progressions.view(states.shape[0], 32, 1, 1)
    probs = base_probabilities * valid_progressions
    return progressions, probs