
from copy import deepcopy
from pathlib import Path
from typing import List, Optional
import torch
import logging
from core.test.tester import TwoPlayerTester
from core.train.trainer import Trainer, TrainerConfig
from core.utils.history import Metric, TrainingMetrics
from core.resnet import TurboZeroResnet
from core.utils.memory import GameReplayMemory
from envs.othello.collector import OthelloCollector
from core.resnet import reset_model_weights



class OthelloTrainer(Trainer):
    def __init__(self,
        config: TrainerConfig,
        collector: OthelloCollector,
        tester: TwoPlayerTester,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        raw_train_config: dict,
        raw_env_config: dict,
        history: TrainingMetrics,
        log_results: bool = True,
        interactive: bool = True,
        run_tag: str = 'othello',
        debug: bool = False
    ):
        super().__init__(
            config=config,
            collector=collector,
            tester=tester,
            model=model,
            optimizer=optimizer,
            device=device,
            raw_train_config=raw_train_config,
            raw_env_config=raw_env_config,
            history=history,
            log_results=log_results,
            interactive=interactive,
            run_tag=run_tag,
            debug=debug
        )
    
    def add_collection_metrics(self, episodes):
        for _ in episodes:
            self.history.add_episode_data({}, log=self.log_results)
            
    def add_epoch_metrics(self):
        pass

 