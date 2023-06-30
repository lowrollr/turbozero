import torch

def fast_weighted_sample(weights, sampler):
    torch.rand(sampler.shape, out=sampler) 
    cum_weights = weights.cumsum(dim=1)
    cum_weights.div_(cum_weights[:, -1:])
    return (sampler < cum_weights).long().argmax(dim=1)
