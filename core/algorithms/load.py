

import torch
from core.algorithms.alphazero import AlphaZero, AlphaZeroConfig
from core.algorithms.baselines.baseline import BaselineConfig
from core.algorithms.baselines.random import RandomBaseline
from core.algorithms.evaluator import EvaluatorConfig
from core.algorithms.lazyzero import LazyZero, LazyZeroConfig
from core.demo.human import HumanEvaluator
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
    
def init_evaluator(algo_type: str, algo_config: dict, env: Env):
    if algo_type == 'lazyzero':
        config = LazyZeroConfig(**algo_config)
        return LazyZero(env, config)
    elif algo_type == 'alphazero':
        config = AlphaZeroConfig(**algo_config)
        return AlphaZero(env, config)
    elif algo_type == 'random':
        config = BaselineConfig(**algo_config)
        return RandomBaseline(env, config)
    elif algo_type == 'human':
        config = EvaluatorConfig(**algo_config)
        return HumanEvaluator(env, config)
    else:
        raise NotImplementedError(f'Unknown evaluator type: {algo_type}')