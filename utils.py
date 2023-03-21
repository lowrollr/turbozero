import torch
import numpy as np

def input_to_tensor(board_states):
    # convert board state values to one-hot encodings
    board_state = np.stack(board_states)
    one_hotified = np.eye(18)[board_state].reshape(-1, 18, 4, 4)
    tensor = torch.from_numpy(one_hotified).float()
    return tensor


def input_to_tensor_3d(board_states):
    board_state = np.stack(board_states)
    one_hotified = np.eye(18)[board_state].reshape(-1, 1, 18, 4, 4)
    tensor = torch.from_numpy(one_hotified).float()
    return tensor