from dataclasses import dataclass
import flax.linen as nn
from flax import struct


@dataclass
class AZResnetConfig:
    model_type: str
    policy_head_out_size: int
    value_head_out_size: int
    num_blocks: int
    num_channels: int

class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x, train: bool):
        y = nn.Conv(features=self.channels, kernel_size=(3,3), strides=(1,1), padding='SAME')(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(features=self.channels, kernel_size=(3,3), strides=(1,1), padding='SAME')(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        return nn.relu(x + y)

class AZResnet(nn.Module):
    config: AZResnetConfig

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=self.config.num_channels, kernel_size=(1,1), strides=(1,1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        for _ in range(self.config.num_blocks):
            x = ResidualBlock(channels=self.config.num_channels)(x, train=train)

        # policy head
        policy = nn.Conv(features=2, kernel_size=(1,1), strides=(1,1), padding='SAME')(x)
        policy = nn.BatchNorm(use_running_average=not train)(policy)
        policy = nn.relu(policy)
        policy = policy.reshape((policy.shape[0], -1))
        policy = nn.Dense(features=self.config.policy_head_out_size)(policy)

        # value head
        value = nn.Conv(features=1, kernel_size=(1,1), strides=(1,1), padding='SAME')(x)
        value = nn.BatchNorm(use_running_average=not train)(value)
        value = nn.relu(value)
        value = value.reshape((value.shape[0], -1))
        value = nn.Dense(features=self.config.value_head_out_size)(value)
        value = nn.tanh(value)

        return policy, value
    