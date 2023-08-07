



import torch

from core.env import Env


def load_baseline(config: dict, env: Env, **kwargs):
    if config['name'] == 'random':
        from core.algorithms.baselines.random import RandomBaseline
        from core.algorithms.baselines.baseline import BaselineConfig
        baseline_config = BaselineConfig(**config)
        return RandomBaseline(env, baseline_config, **kwargs)
    elif config['name']== 'best':
        from core.algorithms.baselines.best import BestModelBaseline, BestModelBaselineConfig
        baseline_config = BestModelBaselineConfig(**config)
        return BestModelBaseline(env, baseline_config, **kwargs)
    else:
        raise ValueError(f'Unknown baseline name: {config["name"]}')
