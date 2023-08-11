import torch


def rand_argmax_2d(values):
    inds = values == values.max(dim=1, keepdim=True).values
    return torch.multinomial(inds.float(), 1)


