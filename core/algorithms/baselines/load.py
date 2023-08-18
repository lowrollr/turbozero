




from core.algorithms.baselines.baseline import Baseline
from core.algorithms.baselines.best import BestModelBaseline
from core.algorithms.baselines.greedy import GreedyBaseline, GreedyConfig
from core.algorithms.baselines.greedy_mcts import GreedyMCTS, GreedyMCTSConfig
from core.algorithms.baselines.lazy_greedy_mcts import LazyGreedyMCTS, LazyGreedyMCTSConfig
from core.algorithms.baselines.random import RandomBaseline
from core.algorithms.baselines.rollout_mcts import RandomRolloutMCTS, RandomRolloutMCTSConfig
from core.algorithms.evaluator import EvaluatorConfig
from core.env import Env


def init_baseline(evaluator_config: dict, env: Env, *args, **kwargs) -> Baseline:
    algo_type = evaluator_config['name']
    
    if algo_type == 'random':
        config = EvaluatorConfig(**evaluator_config)
        return RandomBaseline(env, config, *args, **kwargs)
    elif algo_type == 'best':
        config = EvaluatorConfig(**evaluator_config)
        return BestModelBaseline(env, config, *args, **kwargs)
    elif algo_type == 'greedy_mcts':
        config = GreedyMCTSConfig(**evaluator_config)
        return GreedyMCTS(env, config, *args, **kwargs)
    elif algo_type == 'greedy':
        config = GreedyConfig(**evaluator_config)
        return GreedyBaseline(env, config, *args, **kwargs)
    elif algo_type == 'random_rollout_mcts':
        config = RandomRolloutMCTSConfig(**evaluator_config)
        return RandomRolloutMCTS(env, config, *args, **kwargs)
    elif algo_type == 'lazy_greedy_mcts':
        config = LazyGreedyMCTSConfig(**evaluator_config)
        return LazyGreedyMCTS(env, config, *args, **kwargs)
    else:
        raise NotImplementedError(f'Unknown evaluator type: {algo_type}')