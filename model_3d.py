import torch.nn as nn
import torch

class MonteCarlo3d(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(2,2,2), stride=1, padding=0), # 1 x 19 x 4 x 4 -> 16 x 18 x 3 x 3
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(18,1,1), stride=1, padding=0), # 16 x 18 x 3 x 3 -> 16 x 18 x 3 x 3
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(2,2,2), stride=1, padding=0), # 16 x 18 x 3 x 3 -> 16 x 17 x 2 x 2
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(15,1,1), stride=1, padding=0), # 16 x 17 x 2 x 2 -> 16 x 17 x 2 x 2
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.ReLU()
        self.lin = nn.Linear((16 * 2 * 2), 64)
        self.value_lin1 = nn.Linear(64, 32)
        self.value_lin2 = nn.Linear(32, 1)
        self.prob_lin1 = nn.Linear(64, 32)
        self.prob_lin2 = nn.Linear(32, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0,0,0,0,0,1))
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[3], x.shape[4])
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.lin(self.flatten(x))
        x_v = self.relu(self.value_lin1(x))
        x_v = self.value_lin2(x_v)
        x_p = self.relu(self.prob_lin1(x))
        x_p = self.softmax(self.prob_lin2(x_p))
        return x_p, x_v
        