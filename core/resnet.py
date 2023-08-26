
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Callable

from core.utils.custom_activations import load_activation


@dataclass
class ResNetConfig:
    res_channels: int
    res_blocks: int
    kernel_size: int
    value_fc_size: int = 32
    value_output_activation: str = ''
    
def reset_model_weights(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = 'same', bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = 'same', bias=False),
                        nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out
    
    def fuse(self):
        torch.quantization.fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        torch.quantization.fuse_modules(self.conv2, ['0', '1'], inplace=True)


class TurboZeroResnet(nn.Module):
    def __init__(self, config: ResNetConfig, input_shape: torch.Size, output_shape: torch.Size) -> None:
        super().__init__()
        assert len(input_shape) == 3  # (channels, height, width)
        self.value_head_activation: Optional[torch.nn.Module] = load_activation(config.value_output_activation)
        self.input_channels, self.input_height, self.input_width = input_shape

        self.input_block = nn.Sequential(
            nn.Conv2d(self.input_channels, config.res_channels, kernel_size = config.kernel_size, stride = 1, padding = 'same', bias=False),
            nn.BatchNorm2d(config.res_channels),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(config.res_channels, config.res_channels, config.kernel_size) \
            for _ in range(config.res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(config.res_channels, 2, kernel_size = 1, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(2 * self.input_height * self.input_width, output_shape[0])
            # we use cross entropy loss so no need for softmax
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(config.res_channels, 1, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(self.input_height * self.input_width, config.value_fc_size),
            nn.ReLU(),
            nn.Linear(config.value_fc_size, 1)
            # value head activation handled in forward
        )

        self.config = config

    def forward(self, x):
        x = self.input_block(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, self.value_head_activation(value) if self.value_head_activation is not None else value
    
    def fuse(self):
        torch.quantization.fuse_modules(self.input_block, ['0', '1', '2'], inplace=True)
        for b in self.res_blocks:
            if isinstance(b, ResidualBlock):
                b.fuse()
        for b in self.policy_head:
            if isinstance(b, ResidualBlock):
                b.fuse()
        for b in self.value_head:
            if isinstance(b, ResidualBlock):
                b.fuse()
        
        
            
