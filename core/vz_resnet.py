
import torch
import torch.nn as nn

from dataclasses import dataclass


@dataclass
class VZArchitectureParameters:
    input_size: torch.Size
    policy_size: int
    res_channels: int
    res_blocks: int
    value_head_res_channels: int
    value_head_res_blocks: int
    policy_head_res_channels: int
    policy_head_res_blocks: int
    kernel_size: int
    policy_fc_size: int = 32
    value_fc_size: int = 32


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = 'same', bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def fuse(self):
        torch.quantization.fuse_modules(self, ['conv', 'bn', 'relu'], inplace=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, stride)
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
        self.conv1.fuse()
        self.conv2.fuse()


class VZResnet(nn.Module):
    def __init__(self, arch_params: VZArchitectureParameters) -> None:
        super().__init__()
        assert len(arch_params.input_size) == 3  # (channels, height, width)
        self.input_channels, self.input_height, self.input_width = arch_params.input_size

        self.input_block = ConvBlock(self.input_channels, arch_params.res_channels, arch_params.kernel_size, stride = 1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(arch_params.res_channels, arch_params.res_channels, arch_params.kernel_size) \
            for _ in range(arch_params.res_blocks)]
        )

        self.policy_head = nn.Sequential(
            *[ResidualBlock(arch_params.res_channels, arch_params.policy_head_res_channels, arch_params.kernel_size) \
            for _ in range(arch_params.policy_head_res_blocks)],
            nn.Flatten(start_dim=1),
            nn.Linear(arch_params.policy_head_res_channels * self.input_height * self.input_width, arch_params.policy_fc_size, bias=False),
            nn.BatchNorm1d(arch_params.policy_fc_size),
            nn.ReLU(),
            nn.Linear(arch_params.policy_fc_size, arch_params.policy_size),
            # we use cross entropy loss so no need for softmax
        )

        self.value_head = nn.Sequential(
            *[ResidualBlock(arch_params.res_channels, arch_params.value_head_res_channels, arch_params.kernel_size) \
            for _ in range(arch_params.value_head_res_blocks)],
            nn.Flatten(start_dim=1),
            nn.Linear(arch_params.value_head_res_channels * self.input_height * self.input_width, arch_params.value_fc_size, bias=False),
            nn.BatchNorm1d(arch_params.value_fc_size),
            nn.ReLU(),
            nn.Linear(arch_params.value_fc_size, 1)
        )

        self.arch_params = arch_params

    def forward(self, x):
        x = self.input_block(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    
    def fuse(self):
        self.input_block.fuse()
        for b in self.res_blocks:
            if isinstance(b, ResidualBlock):
                b.fuse()
        for b in self.policy_head:
            if isinstance(b, ResidualBlock):
                b.fuse()
        for b in self.value_head:
            if isinstance(b, ResidualBlock):
                b.fuse()
