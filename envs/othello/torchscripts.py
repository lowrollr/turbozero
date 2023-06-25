import torch


def get_legal_actions(states, ray_tensor):
    states_size = states.shape[-1]
    index = 1
    for i in range(2, states_size):
        start, end = 0, states_size - i 
        start_inv, end_inv = i, states_size
        kernel_right = torch.zeros((1, 2, 1, i+1), device=states.device, requires_grad=False, dtype=torch.long)
        kernel_right[:, 1, :, 1:i] = 1
        kernel_right[:, 0, :, i] = 1
        kernel_right[:, :, :, 0] = -1
        ray_tensor[:, index, :, start:end] = torch.nn.functional.conv2d(states, kernel_right, padding=0).squeeze(1) == i
        index += 1

        kernel_down = torch.zeros((1, 2, i+1, 1), device=states.device, requires_grad=False, dtype=torch.long)
        kernel_down[:, 1, 1:i, :] = 1
        kernel_down[:, 0, i, :] = 1
        kernel_down[:, :, 0, :] = -1
        ray_tensor[:, index, start:end, :] = torch.nn.functional.conv2d(states, kernel_down, padding=0).squeeze(1) == i
        index += 1

        kernel_left = torch.zeros((1, 2, 1, i+1), device=states.device, requires_grad=False, dtype=torch.long)
        kernel_left[:, 1, :, 1:-1] = 1
        kernel_left[:, 0, :, 0] = 1
        kernel_left[:, :, :, -1] = -1
        ray_tensor[:, index, :, start_inv:end_inv] = torch.nn.functional.conv2d(states, kernel_left, padding=0).squeeze(1) == i
        index += 1

        kernel_up = torch.zeros((1, 2, i+1, 1), device=states.device, requires_grad=False, dtype=torch.long)
        kernel_up[:, 1, 1:-1, :] = 1
        kernel_up[:, 0, 0, :] = 1
        kernel_up[:, :, -1, :] = -1
        ray_tensor[:, index, start_inv:end_inv, :] = torch.nn.functional.conv2d(states, kernel_up, padding=0).squeeze(1) == i
        index += 1

        kernel_diag_right_down = torch.zeros((1, 2, i+1, i+1), device=states.device, requires_grad=False, dtype=torch.long) 
        for j in range(1, i):
            kernel_diag_right_down[:, 1, j, j] = 1
        kernel_diag_right_down[:, 0, i, i] = 1
        kernel_diag_right_down[:, :, 0, 0] = -1
        ray_tensor[:, index, start:end, start:end] = torch.nn.functional.conv2d(states, kernel_diag_right_down, padding=0).squeeze(1) == i
        index += 1

        kernel_diag_left_down = torch.zeros((1, 2, i+1, i+1), device=states.device, requires_grad=False, dtype=torch.long)
        for j in range(1, i):
            kernel_diag_left_down[:, 1, j, -j-1] = 1
        kernel_diag_left_down[:, 0, i, 0] = 1
        kernel_diag_left_down[:, :, 0, -1] = -1
        ray_tensor[:, index, start:end, start_inv:end_inv] = torch.nn.functional.conv2d(states, kernel_diag_left_down, padding=0).squeeze(1) == i
        index += 1

        kernel_diag_left_up = torch.zeros((1, 2, i+1, i+1), device=states.device, requires_grad=False, dtype=torch.long)
        for j in range(1, i):
            kernel_diag_left_up[:, 1, -j-1, j] = 1
        kernel_diag_left_up[:, 0, 0, i] = 1
        kernel_diag_left_up[:, :, -1, 0] = -1
        ray_tensor[:, index, start_inv:end_inv, start_inv:end_inv] = torch.nn.functional.conv2d(states, kernel_diag_left_up, padding=0).squeeze(1) == i
        index += 1

        kernel_diag_right_up = torch.zeros((1, 2, i+1, i+1), device=states.device, requires_grad=False, dtype=torch.long)
        for j in range(1, i):
            kernel_diag_right_up[:, 1, -j-1, -j-1] = 1
        kernel_diag_right_up[:, 0, 0, 0] = 1
        kernel_diag_right_up[:, :, -1, -1] = -1
        ray_tensor[:, index, start_inv:end_inv, start:end] = torch.nn.functional.conv2d(states, kernel_diag_right_up, padding=0).squeeze(1) == i
        index += 1

    return ray_tensor.any(dim=1)


def push_actions(states, ray_tensor, actions):
    num_rays = ray_tensor.shape[1]
    states_size = states.shape[-1]
    num_states = states.shape[0]

    flips = torch.zeros((num_rays+1, states_size, states_size, states_size, states_size), device=states.device, requires_grad=False, dtype=torch.long)
    f_index = 1
    for i in range(2, states_size):
        for x in range(states_size):
            for y in range(states_size):
                # right, down, left, up
                if x+1 < states_size:
                    flips[f_index, y, x, y, x+1:min(x+i, states_size)] = 1
                if y+1 < states_size:
                    flips[f_index+1, y, x, y+1:min(y+i, states_size), x] = 1
                flips[f_index+2, y, x, y, max(x-i, 0):x] = 1
                flips[f_index+3, y, x, max(y-i, 0):y, x] = 1

                # diag right down, diag left down, diag left up, diag right up
                for j in range(1, i):
                    if y+j < states_size:
                        if x+j < states_size:
                            flips[f_index+4, y, x, y+j, x+j] = 1
                        if x-j >= 0:
                            flips[f_index+5, y, x, y+j, x-j] = 1
                    if y-j >= 0:
                        if x-j >= 0:
                            flips[f_index+6, y, x, y-j, x-j] = 1
                        if x+j < states_size:
                            flips[f_index+7, y, x, y-j, x+j] = 1
        f_index += 8

    action_ys, action_xs = actions // states_size, actions % states_size

    activated_rays = ray_tensor[torch.arange(num_states), :, action_ys, action_xs] * (torch.arange(num_rays, device=states.device).unsqueeze(0).repeat(num_states, 1))

    flips_to_apply = flips[activated_rays, action_ys.unsqueeze(1).repeat(1, 49), action_xs.unsqueeze(1).repeat(1, 49)].amax(dim=1)

    not_pass = ~states[torch.arange(num_states), :, action_ys, action_xs].any(dim=1).view(-1, 1, 1)
    states[:, 0, :, :] |= (flips_to_apply * not_pass)
    states[:, 1, :, :] &= ~(flips_to_apply * not_pass)
    states[torch.arange(num_states), 0, action_ys, action_xs] = not_pass.view(-1).long()
    return states