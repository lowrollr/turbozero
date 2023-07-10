


import torch


class Tanh0to1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):
        return (self.tanh(x) + 1) / 2
