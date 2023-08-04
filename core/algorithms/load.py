

import torch
from core.algorithms.alphazero import AlphaZero, AlphaZeroConfig
from core.algorithms.lazyzero import LazyZero, LazyZeroConfig
from core.env import Env


def init_trainable_evaluator(algo_type: str, algo_config: dict, env: Env):
    if algo_type == 'lazyzero':
        config = LazyZeroConfig(**algo_config)
        return LazyZero(env, config)
    elif algo_type == 'alphazero':
        config = AlphaZeroConfig(**algo_config)
        return AlphaZero(env, config)
    else:
        raise NotImplementedError(f'Unknown trainable evaluator type: {algo_type}')