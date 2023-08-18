import torch


def rand_argmax_2d(values):
    inds = values == values.max(dim=1, keepdim=True).values
    return torch.multinomial(inds.float(), 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)