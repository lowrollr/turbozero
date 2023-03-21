import torch.nn as nn
import torch

class MonteCarlo3d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(2,2,2), stride=(1,1,1), padding=(0, 0, 0)), # 1 x 18 x 4 x 4 -> 16 x 17 x 3 x 3
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(2,2,2), stride=(1,1,1), padding=(0,0,0)), # 16 x 17 x 3 x 3 -> 16 x 16 x 2 x 2
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.ReLU()
        self.lin = nn.Linear((16 * 17 * 3 * 3) + (16 * 16 * 2 * 2), 256)
        self.value_lin1 = nn.Linear(256, 128)
        self.value_lin2 = nn.Linear(128, 1)
        self.prob_lin1 = nn.Linear(256, 128)
        self.prob_lin2 = nn.Linear(128, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        x = self.relu(self.lin(torch.cat([self.flatten(x1), self.flatten(x2)], dim=1)))
        x_v = self.relu(self.value_lin1(x))
        x_v = self.value_lin2(x_v)
        x_p = self.relu(self.prob_lin1(x))
        x_p = self.softmax(self.prob_lin2(x_p))
        return x_p, x_v
        