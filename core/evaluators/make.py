




from core.envs.env import Env


def make_evaluator(evaluator_config: dict, env: Env, **kwargs):
    evaluator_type = evaluator_config.get('evaluator_type')

    if evaluator_type == 'alphazero':
        from core.evaluators.alphazero import AlphaZero, AlphaZeroConfig
        config = AlphaZeroConfig(**evaluator_config)
        return AlphaZero(env, config, **kwargs)
    elif evaluator_type == 'randotron':
        from core.evaluators.randotron import RandotronEvaluator, RandotronEvaluatorConfig
        config = RandotronEvaluatorConfig(**evaluator_config)
        return RandotronEvaluator(env, config, **kwargs)
    elif evaluator_type == '':
        raise NotImplementedError('Evaluator type not specified in config')
    else:
        raise NotImplementedError(f'Unknown evaluator type {evaluator_type}')
