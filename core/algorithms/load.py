

import torch
from core.algorithms.alphazero import AlphaZero, AlphaZeroConfig
from core.algorithms.baselines.greedy import GreedyBaseline
from core.algorithms.baselines.greedy_mcts import GreedyMCTS
from core.algorithms.baselines.random import RandomBaseline
from core.algorithms.baselines.rollout_mcts import RandomRolloutMCTS, RandomRolloutMCTSConfig
from core.algorithms.evaluator import EvaluatorConfig
from core.algorithms.lazy_mcts import LazyMCTSConfig
from core.algorithms.lazyzero import LazyZero, LazyZeroConfig
from core.algorithms.mcts import MCTSConfig
# from core.demo.human import HumanEvaluator
from core.env import Env


def init_trainable_evaluator(algo_config: dict, env: Env, model: torch.nn.Module, *args, **kwargs):
    algo_type = algo_config['name']
    if algo_type == 'lazyzero':
        config = LazyZeroConfig(**algo_config)
        return LazyZero(env, config, model, *args, **kwargs)
    elif algo_type == 'alphazero':
        config = AlphaZeroConfig(**algo_config)
        return AlphaZero(env, config, model, *args, **kwargs)
    else:
        raise NotImplementedError(f'Unknown trainable evaluator type: {algo_type}')
    
def init_evaluator(algo_config: dict, env: Env, *args, **kwargs):
    algo_type = algo_config['name']
    if algo_type == 'lazyzero':
        config = LazyZeroConfig(**algo_config)
        return LazyZero(env, config, *args, **kwargs)
    elif algo_type == 'alphazero':
        config = AlphaZeroConfig(**algo_config)
        return AlphaZero(env, config,  *args, **kwargs)
    elif algo_type == 'random':
        config = EvaluatorConfig(**algo_config)
        return RandomBaseline(env, config, *args, **kwargs)
    # elif algo_type == 'human':
    #     config = EvaluatorConfig(**algo_config)
    #     return HumanEvaluator(env, config, *args, **kwargs)
    elif algo_type == 'greedy_mcts':
        config = MCTSConfig(**algo_config)
        return GreedyMCTS(env, config, *args, **kwargs)
    elif algo_type == 'greedy':
        config = EvaluatorConfig(**algo_config)
        return GreedyBaseline(env, config, *args, **kwargs)
    elif algo_type == 'random_rollout_mcts':
        config = RandomRolloutMCTSConfig(**algo_config)
        return RandomRolloutMCTS(env, config, *args, **kwargs)
    else:
        raise NotImplementedError(f'Unknown evaluator type: {algo_type}')
    
