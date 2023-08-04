


from typing import Optional
import torch
import logging


class Tanh0to1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):
        return (self.tanh(x) + 1) / 2


def load_activation(activation: str) -> Optional[torch.nn.Module]:
    if activation == 'tanh0to1':
        return Tanh0to1()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    else:
        if activation != '':
            logging.warn(f'Warning: activation {activation} not found')
            logging.warn(f'No activation will be applied to value head')
        return None