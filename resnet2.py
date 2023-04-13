import torch.nn as nn
import torch

# https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = stride, padding = 'same', bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 2, stride = 1, padding = 'same', bias=False),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet2Heads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(2,2), stride=1, padding='same', bias=False), # 1 x 19 x 4 x 4 -> 16 x 18 x 3 x 3
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2,2), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 3 * 3, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 4),
            # nn.Softmax(dim=1) using cross entropy so just need logits
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2,2), stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 3 * 3, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
            

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value