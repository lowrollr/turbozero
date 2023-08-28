import torch


def rand_argmax_2d(values):
    inds = values == values.max(dim=1, keepdim=True).values
    return torch.multinomial(inds.float(), 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def jaccard_similarity(tensor1, tensor2):
    # Compute the intersection and union
    intersection = (tensor1 & tensor2).float().sum()
    union = (tensor1 | tensor2).float().sum()
    
    # Compute the Jaccard similarity coefficient
    jaccard = intersection / union
    
    return jaccard


def jaccard_centroid_similarity(data):
    
    # Compute the centroid of the dataset
    centroid = data.float().mean(dim=0).bool()
    
    # Compute the intersection and union
    intersection = (data.bool() & centroid).float().sum(dim=(1, 2))
    union = (data.bool() | centroid).float().sum(dim=(1, 2))
    
    # Compute the Jaccard similarities
    jaccard_similarities = intersection / union
    
    return jaccard_similarities.mean().item()