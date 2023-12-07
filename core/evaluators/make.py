from core.envs.env import Env
from core.evaluators.evaluator import Evaluator


def make_evaluator(
    evaluator_type: str,
    config: dict, 
    env: Env, 
    **kwargs
) -> Evaluator:
    if evaluator_type == 'alphazero':
        from core.evaluators.alphazero import AlphaZero, AlphaZeroConfig
        config = AlphaZeroConfig(evaluator_type=evaluator_type, **config)
        return AlphaZero(env, config, **kwargs)
    elif evaluator_type == 'randotron':
        from core.evaluators.randotron import RandotronEvaluator, RandotronEvaluatorConfig
        config = RandotronEvaluatorConfig(evaluator_type=evaluator_type, **config)
        return RandotronEvaluator(env, config, **kwargs)
    elif evaluator_type == '':
        raise NotImplementedError('Evaluator type not specified in config')
    else:
        raise NotImplementedError(f'Unknown evaluator type {evaluator_type}')
