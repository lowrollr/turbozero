import torch.nn as nn
import torch 
import numpy as np
# Network Architecture

# Given a state (4 x 4 x 12), we want to predict the Q-value for each action. 
class DeepMonteCarlo(torch.nn.Module):
    def __init__(self, input_channels, ) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(2,2), stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2,2), stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=(2,2), stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(2,2), stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        self.value_fc1 = nn.Linear(16 * 4 * 4, 16 * 4 * 4)
        self.value_fc2 = nn.Linear(16 * 4 * 4, 16)
        self.value_fc3 = nn.Linear(16, 1)
        self.prob_fc1 = nn.Linear(16 * 4 * 4, 16 * 4 * 4)
        self.prob_fc2 = nn.Linear(16 * 4 * 4, 16)
        self.prob_fc3 = nn.Linear(16, 4)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x_v = self.relu(self.value_fc1(x))
        x_v = self.relu(self.value_fc2(x_v))
        x_v = self.value_fc3(x_v)

        x_p = self.relu(self.prob_fc1(x))
        x_p = self.relu(self.prob_fc2(x_p))
        x_p = self.softmax(self.prob_fc3(x_p))
        return x_p, x_v