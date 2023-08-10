




from core.algorithms.baselines.baseline import Baseline
from core.algorithms.baselines.best import BestModelBaseline
from core.algorithms.baselines.greedy import GreedyBaseline
from core.algorithms.baselines.greedy_mcts import GreedyMCTS
from core.algorithms.baselines.random import RandomBaseline
from core.algorithms.baselines.rollout_mcts import RandomRolloutMCTS, RandomRolloutMCTSConfig
from core.algorithms.evaluator import EvaluatorConfig
from core.algorithms.mcts import MCTSConfig
from core.demo.human import HumanEvaluator
from core.env import Env


def init_baseline(evaluator_config: dict, env: Env, **kwargs) -> Baseline:
    algo_type = evaluator_config['name']
    
    if algo_type == 'random':
        config = EvaluatorConfig(**evaluator_config)
        return RandomBaseline(env, config)
    elif algo_type == 'best':
        config = EvaluatorConfig(**evaluator_config)
        return BestModelBaseline(env, config, kwargs['evaluator'], kwargs['best_model'], kwargs['best_model_optimizer'])
    elif algo_type == 'greedy_mcts':
        config = MCTSConfig(**evaluator_config)
        return GreedyMCTS(env, config)
    elif algo_type == 'greedy':
        config = EvaluatorConfig(**evaluator_config)
        return GreedyBaseline(env, config)
    elif algo_type == 'random_rollout_mcts':
        config = RandomRolloutMCTSConfig(**evaluator_config)
        return RandomRolloutMCTS(env, config)
    else:
        raise NotImplementedError(f'Unknown evaluator type: {algo_type}')