

from envs.othello.evaluators.edax import Edax, EdaxConfig
import torch
from core.algorithms.alphazero import AlphaZero, AlphaZeroConfig
from core.algorithms.baselines.greedy import GreedyBaseline, GreedyConfig
from core.algorithms.baselines.greedy_mcts import GreedyMCTS, GreedyMCTSConfig
from core.algorithms.baselines.lazy_greedy_mcts import LazyGreedyMCTS, LazyGreedyMCTSConfig
from core.algorithms.baselines.lazy_rollout_mcts import RandomRolloutLazyMCTS, RandomRolloutLazyMCTSConfig
from core.algorithms.baselines.random import RandomBaseline
from core.algorithms.baselines.rollout import Rollout, RolloutConfig
from core.algorithms.baselines.rollout_mcts import RandomRolloutMCTS, RandomRolloutMCTSConfig
from core.algorithms.evaluator import EvaluatorConfig
from core.algorithms.lazyzero import LazyZero, LazyZeroConfig
from core.demo.human import HumanEvaluator
from core.env import Env


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
    elif algo_type == 'human':
        config = EvaluatorConfig(**algo_config)
        return HumanEvaluator(env, config, *args, **kwargs)
    elif algo_type == 'greedy_mcts':
        config = GreedyMCTSConfig(**algo_config)
        return GreedyMCTS(env, config, *args, **kwargs)
    elif algo_type == 'edax':
        config = EdaxConfig(**algo_config)
        return Edax(env, config, *args, **kwargs)
    elif algo_type == 'greedy':
        config = GreedyConfig(**algo_config)
        return GreedyBaseline(env, config, *args, **kwargs)
    elif algo_type == 'random_rollout_mcts':
        config = RandomRolloutMCTSConfig(**algo_config)
        return RandomRolloutMCTS(env, config, *args, **kwargs)
    elif algo_type == 'random_rollout_lazy_mcts':
        config = RandomRolloutLazyMCTSConfig(**algo_config)
        return RandomRolloutLazyMCTS(env, config, *args, **kwargs)
    elif algo_type == 'rollout':
        config = RolloutConfig(**algo_config)
        return Rollout(env, config, *args, **kwargs)
    elif algo_type == 'lazy_greedy_mcts':
        config = LazyGreedyMCTSConfig(**algo_config)
        return LazyGreedyMCTS(env, config, *args, **kwargs)
    else:
        raise NotImplementedError(f'Unknown evaluator type: {algo_type}')
    
