
from dataclasses import dataclass
import logging
import torch
from typing import Optional, Tuple, Union
from core.algorithms.evaluator import Evaluator

from core.env import EnvConfig
from core.resnet import TurboZeroResnet
from core.test.tester import TesterConfig, Tester, TwoPlayerTesterConfig, TwoPlayerTester
from core.train.collector import Collector
from core.train.trainer import Trainer, TrainerConfig
from core.utils.history import TrainingMetrics
from envs._2048.collector import _2048Collector
from envs._2048.tester import _2048Tester
from envs._2048.trainer import _2048Trainer
from envs.othello.collector import OthelloCollector
from envs.othello.tester import OthelloTester
from envs.othello.trainer import OthelloTrainer
from .othello.env import OthelloEnv, OthelloEnvConfig
from ._2048.env import _2048Env, _2048EnvConfig

def init_env(device: torch.device, parallel_envs: int, env_config: dict, debug: bool):
    env_type = env_config['env_type']
    if env_type == 'othello':
        config = OthelloEnvConfig(**env_config)
        return OthelloEnv(parallel_envs, config, device, debug)
    elif env_type == '2048':
        config = _2048EnvConfig(**env_config)
        return _2048Env(parallel_envs, config, device, debug)
    else:
        raise NotImplementedError(f'Environment {env_type} not implemented')
    
def init_collector(episode_memory_device: torch.device, env_type: str, evaluator: Evaluator):
    if env_type == 'othello':
        return OthelloCollector(
            evaluator=evaluator,
            episode_memory_device=episode_memory_device
        )
    elif env_type == '2048':
        return _2048Collector(
            evaluator=evaluator,
            episode_memory_device=episode_memory_device
        )
    else:
        raise NotImplementedError(f'Collector for environment {env_type} not supported')
    
def init_tester(
    test_config: dict,
    collector: Collector,
    model: torch.nn.Module,
    history: TrainingMetrics,
    optimizer: Optional[torch.optim.Optimizer],
    log_results: bool,
    debug: bool
):
    if collector.evaluator.env.num_players == 2:
        return OthelloTester(
            config=TwoPlayerTesterConfig(**test_config),
            collector=collector,
            model=model,
            optimizer=optimizer,
            history=history,
            log_results=log_results,
            debug=debug
        )
    elif collector.evaluator.env.num_players == 1:
        return _2048Tester(
            config=TesterConfig(**test_config),
            collector=collector,
            model=model,
            optimizer=optimizer,
            history=history,
            log_results=log_results,
            debug=debug
        )
    else:
        raise NotImplementedError(f'Tester for {collector.evaluator.env.num_players} players not supported')

def init_trainer(
    device: torch.device, 
    env_type: str, 
    collector: Collector, 
    tester: Tester, 
    model: TurboZeroResnet,
    optimizer: torch.optim.Optimizer,
    train_config: dict,
    raw_env_config: dict,
    history: TrainingMetrics,
    log_results: bool,
    interactive: bool,
    run_tag: str = '',
    debug: bool = False
):
    trainer_config = TrainerConfig(**train_config)
    if env_type == 'othello':
        assert isinstance(collector, OthelloCollector)
        assert isinstance(tester, TwoPlayerTester)
        return OthelloTrainer(
            config = trainer_config,
            collector = collector,
            tester = tester,
            model = model,
            optimizer = optimizer,
            device = device,
            raw_train_config = train_config,
            raw_env_config = raw_env_config,
            history = history,
            log_results=log_results,
            interactive=interactive,
            run_tag = run_tag,
            debug = debug
        )
    elif env_type == '2048':
        assert isinstance(collector, _2048Collector)
        return _2048Trainer(
            config = trainer_config,
            collector = collector,
            tester = tester,
            model = model,
            optimizer = optimizer,
            device = device,
            raw_train_config = train_config,
            raw_env_config = raw_env_config,
            history = history,
            log_results=log_results,
            interactive=interactive,
            run_tag = run_tag,
            debug = debug
        )
    else:
        logging.warn(f'No trainer found for environment {env_type}')
        return Trainer(
            config = trainer_config,
            collector = collector,
            tester = tester,
            model = model,
            optimizer = optimizer,
            device = device,
            raw_train_config = train_config,
            raw_env_config = raw_env_config,
            history = history,
            log_results=log_results,
            interactive=interactive,
            run_tag = run_tag,
            debug = debug
        )
            