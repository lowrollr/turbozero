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


def cosine_centroid_similarity(data):
    centroid = data.mean(dim=0)
    
    centroid_similarity = torch.nn.functional.cosine_similarity(data, centroid.unsqueeze(0), dim=1).mean().item()
    
    return centroid_similarity