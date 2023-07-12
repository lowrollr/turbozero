import torch


def build_filters(device: torch.device, board_size: int):
    num_filters = (board_size - 2) * 8
    filters = torch.zeros((num_filters + 1, 2, board_size, board_size), dtype=torch.float32, device=device, requires_grad=False)

    index = 1
    top_left_indices = []
    top_right_indices = []
    bottom_left_indices = []
    bottom_right_indices = []
    close_to = (torch.arange(-1, num_filters, dtype=torch.long, device=device) // 8) + 2
    
    for i in range(2, board_size):
        # right
        filters[index, 1, 0, 1:i] = 1
        filters[index, 0, 0, i] = 1
        filters[index, :, 0, 0] = -1
        top_left_indices.append(index)
        index += 1
        # left
        filters[index, 1, -1, -i:-1] = 1
        filters[index, 0, -1, -i-1] = 1
        filters[index, :, -1, -1] = -1
        bottom_right_indices.append(index)
        index += 1
        # down
        filters[index, 1, 1:i, 0] = 1
        filters[index, 0, i, 0] = 1
        filters[index, :, 0, 0] = -1
        top_left_indices.append(index)
        index += 1
        # up
        filters[index, 1, -i:-1, -1] = 1
        filters[index, 0, -i-1, -1] = 1
        filters[index, :, -1, -1] = -1
        bottom_right_indices.append(index)
        index += 1

        for j in range(1, i):
            filters[index, 1, j, j] = 1 # down right
            filters[index+1, 1, -j-1, j] = 1 # up right
            filters[index+2, 1, -j-1, -j-1] = 1 # up left
            filters[index+3, 1, j, -j-1] = 1 # down left

        # down right
        filters[index, 0, i, i] = 1 
        filters[index, :, 0, 0] = -1
        top_left_indices.append(index)
        index += 1
        # up right
        filters[index, 0, -1-i, i] = 1 
        filters[index, :, -1, 0] = -1
        bottom_left_indices.append(index)
        index += 1
        # up left
        filters[index, 0, -1-i, -1-i] = 1
        filters[index, :, -1, -1] = -1
        bottom_right_indices.append(index)
        index += 1
        # down left
        filters[index, 0, i, -1-i] = 1
        filters[index, :, 0, -1] = -1
        top_right_indices.append(index)
        index += 1
    

    return filters, \
        torch.tensor(bottom_left_indices, device=device, requires_grad=False), \
        torch.tensor(bottom_right_indices, device=device, requires_grad=False), \
        torch.tensor(top_left_indices, device=device, requires_grad=False), \
        torch.tensor(top_right_indices, device=device, requires_grad=False), \
        close_to.view(1, -1, 1, 1)

def build_flips(num_rays, states_size, device):
    flips = torch.zeros((num_rays, states_size, states_size, states_size, states_size), device=device, requires_grad=False, dtype=torch.float32)
    f_index = 1
    for i in range(2, states_size):
        for x in range(states_size):
            for y in range(states_size):
                # right, left, down, up
                if x+1 < states_size:
                    flips[f_index, y, x, y, x+1:min(x+i, states_size)] = 1
                flips[f_index+1, y, x, y, max(x-i+1, 0):x] = 1
                if y+1 < states_size:
                    flips[f_index+2, y, x, y+1:min(y+i, states_size), x] = 1
                flips[f_index+3, y, x, max(y-i+1, 0):y, x] = 1

                # diag right down, diag left down, diag left up, diag right up
                for j in range(1, i):
                    if y+j < states_size:
                        if x+j < states_size:
                            flips[f_index+4, y, x, y+j, x+j] = 1
                        if x-j >= 0:
                            flips[f_index+7, y, x, y+j, x-j] = 1
                    if y-j >= 0:
                        if x-j >= 0:
                            flips[f_index+6, y, x, y-j, x-j] = 1
                        if x+j < states_size:
                            flips[f_index+5, y, x, y-j, x+j] = 1
        f_index += 8
    return flips



def get_legal_actions(states, ray_tensor, legal_actions, filters, bl_idx, br_idx, tl_idx, tr_idx, ct):
    board_size = int(states.shape[-1]) # need to wrap in int() for tracing
    conv_results = torch.nn.functional.conv2d(states, filters, padding=board_size-1, bias=None)
    ray_tensor.zero_()
    ray_tensor[:, tl_idx] = conv_results[:, tl_idx, board_size-1:, board_size-1:]
    ray_tensor[:, tr_idx] = conv_results[:, tr_idx, board_size-1:, :-(board_size-1)]
    ray_tensor[:, bl_idx] = conv_results[:, bl_idx, :-(board_size-1), board_size-1:]
    ray_tensor[:, br_idx] = conv_results[:, br_idx, :-(board_size-1), :-(board_size-1)]
    ray_tensor[:] = (ray_tensor.round() == ct).float()
    legal_actions.zero_()
    legal_actions[:,:board_size**2] = ray_tensor.any(dim=1).view(-1, board_size ** 2)
    legal_actions[:,board_size**2] = ~(legal_actions.any(dim=1))
    return legal_actions


def push_actions(states, ray_tensor, actions, flips):
    num_rays = ray_tensor.shape[1]
    states_size = states.shape[-1]
    num_states = states.shape[0]
    state_indices = torch.arange(num_states, device=states.device, requires_grad=False, dtype=torch.long)

    is_not_null = actions != states_size ** 2
    action_ys, action_xs = actions // states_size, actions % states_size
    action_ys *= is_not_null # puts null action in-bounds
    action_xs *= is_not_null
    activated_rays = ray_tensor[state_indices, :, action_ys, action_xs] * (torch.arange(num_rays, device=states.device, requires_grad=False).unsqueeze(0)) * is_not_null.view(-1, 1)

    flips_to_apply = flips[activated_rays.long(), action_ys.unsqueeze(1), action_xs.unsqueeze(1)].amax(dim=1) * is_not_null.view(-1, 1, 1)

    states[:, 0, :, :].logical_or_(flips_to_apply)
    states[:, 1, :, :] *= torch.logical_not(flips_to_apply)
    states[state_indices, 0, action_ys, action_xs] += is_not_null.float()
    return states, ~is_not_null