import torch
import numpy as np

def input_to_tensor(board_states):
    # convert board state values to one-hot encodings
    board_state = np.stack(board_states)
    one_hotified = np.eye(18)[board_state]
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