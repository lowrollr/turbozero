import torch
import numpy as np
import numba

def input_to_tensor(board_states):
    # convert board state values to one-hot encodings
    board_state = np.stack(board_states)
    one_hotified = np.eye(18)[board_state].transpose(0, 3, 1,2)
    tensor = torch.from_numpy(one_hotified).float()
    return tensor

# encodes board states as one-hot tensors
# 4 x 4 board state -> 18 x 4 x 4 tensor
def input_to_tensor_3d(board_states, num_classes=18):
    # convert board state values to one-hot encodings
    board_state = np.stack(board_states, axis=0)
    one_hotified = np.eye(num_classes)[board_state].transpose(0, 3, 1,2)
    tensor = torch.from_numpy(one_hotified).float().unsqueeze(1)
    # add a dimension to the front
    return tensor

def input_to_tensor_scalar(board_states):
    tensors = []
    for board in board_states:
        board = np.array(board)
        tensor = torch.stack([
            torch.from_numpy(np.equal(board, 0)).float(),
            *torch.from_numpy(np.stack(compare_tiles(board), axis=0)).float(),
            torch.from_numpy(board).float()
        ], dim=0)
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)

    

# @numba.njit(nogil=True, fastmath=True)
def compare_tiles(arr): # thanks GPT-4!
    # Compare with the tile below
    shifted_down = np.roll(arr, -1, axis=0)
    vertical_comparison = np.logical_and(np.not_equal(arr, 0), np.equal(arr, shifted_down)).astype(float)
    vertical_comparison[:-1, :] = vertical_comparison[1:, :]
    vertical_comparison[-1, :] = 0

    # Compare with the tile to the right
    shifted_right = np.roll(arr, -1, axis=1)
    horizontal_comparison = np.logical_and(np.not_equal(arr, 0), np.equal(arr, shifted_right)).astype(float)
    horizontal_comparison[:, :-1] = horizontal_comparison[:, 1:]
    horizontal_comparison[:, -1] = 0

    return vertical_comparison, horizontal_comparison