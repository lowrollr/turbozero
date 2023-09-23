

import torch
from core.algorithms.load import init_evaluator

from core.demo.demo import Demo
from core.utils.checkpoint import load_checkpoint, load_model_and_optimizer_from_checkpoint
from envs.connect_x.demo import ConnectXDemo
from envs.load import init_env
from envs.othello.demo import OthelloDemo


def init_demo(env_config: dict, demo_config: dict, device: torch.device, *args, **kwargs) -> Demo:
    env = init_env(device, 1, env_config, debug=False)
    evaluator1_config = demo_config['evaluator1_config']
    if evaluator1_config.get('checkpoint'):
        model, _ = load_model_and_optimizer_from_checkpoint(load_checkpoint(evaluator1_config['checkpoint']), env, device)
        evaluator1 = init_evaluator(evaluator1_config['algo_config'], env, model, *args, **kwargs)
    else:
        evaluator1 = init_evaluator(evaluator1_config['algo_config'], env, *args, **kwargs)
    if env_config['env_type'] == 'othello':
        evaluator2_config = demo_config['evaluator2_config']
        if evaluator2_config.get('checkpoint'):
            model, _ = load_model_and_optimizer_from_checkpoint(load_checkpoint(evaluator2_config['checkpoint']), env, device)
            evaluator2 = init_evaluator(evaluator2_config['algo_config'], env, model, *args, **kwargs)
        else:
            evaluator2 = init_evaluator(evaluator2_config['algo_config'], env, *args, **kwargs)
        return OthelloDemo(evaluator1, evaluator2, demo_config['manual_step'])
    elif env_config['env_type'] == 'connect_x':
        evaluator2_config = demo_config['evaluator2_config']
        if evaluator2_config.get('checkpoint'):
            model, _ = load_model_and_optimizer_from_checkpoint(load_checkpoint(evaluator2_config['checkpoint']), env, device)
            evaluator2 = init_evaluator(evaluator2_config['algo_config'], env, model, *args, **kwargs)
        else:
            evaluator2 = init_evaluator(evaluator2_config['algo_config'], env, *args, **kwargs)
        return ConnectXDemo(evaluator1, evaluator2, demo_config['manual_step'])
    elif env_config['env_type'] == '2048':
        return Demo(evaluator1, demo_config['manual_step'])
    else:
        return Demo(evaluator1, demo_config['manual_step'])