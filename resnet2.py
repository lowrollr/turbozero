import torch.nn as nn
import torch

# https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 'same', bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 'same', bias=False),
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
    
class ResNet2Heads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3,3), stride=1, padding='same', bias=False), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(16, 16) for _ in range(8)]
        )

        self.policy_head = nn.Sequential(
            ResidualBlock(16, 16),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * 4 * 4, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 4),
            # nn.Softmax(dim=1) using cross entropy so just need logits
        )

        self.value_head = nn.Sequential(
            ResidualBlock(16, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * 4 * 4, 32, bias=False),
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