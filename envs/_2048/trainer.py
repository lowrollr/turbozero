


from typing import Optional
import torch
import numpy as np
from core.test.tester import Tester
from core.utils.history import Metric, TrainingMetrics
from core.train.trainer import Trainer, TrainerConfig
from envs._2048.collector import _2048Collector

class _2048Trainer(Trainer):
    def __init__(self,
        config: TrainerConfig,
        collector: _2048Collector,
        tester: Tester,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        raw_train_config: dict,
        raw_env_config: dict,
        history: TrainingMetrics,
        log_results: bool = True,
        interactive: bool = True,
        run_tag: str = '2048',
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
        if self.history.cur_epoch == 0:
            self.history.episode_metrics.update({
                'reward': Metric(name='reward', xlabel='Episode', ylabel='Reward', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results),
                'log2_high_square': Metric(name='log2_high_square', xlabel='Episode', ylabel='High Tile', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results, proper_name='High Tile (log2)'),
            })

            self.history.eval_metrics.update({
                'reward': [Metric(name='reward', xlabel='Reward', ylabel='Frequency', pl_type='hist',  maximize=True, alert_on_best=False)],
                'high_square': [Metric(name='high_square', xlabel='High Tile', ylabel='Frequency Tile', pl_type='bar', maximize=True, alert_on_best=False, proper_name='High Tile')]
            })
        
            self.history.epoch_metrics.update({
                'avg_reward': Metric(name='avg_reward', xlabel='Epoch', ylabel='Average Reward', maximize=True, alert_on_best=self.log_results, proper_name='Average Reward'),
                'avg_log2_high_square': Metric(name='avg_log2_high_square', xlabel='Epoch', ylabel='Average High Tile', maximize=True, alert_on_best=self.log_results, proper_name='Average High Tile (log2)'),
            })
    
    def add_collection_metrics(self, episodes):
        for episode in episodes:
            moves = len(episode)
            last_state = episode[-1][0]
            high_square = int(last_state.max().item())
            self.history.add_episode_data({
                'reward': moves,
                'log2_high_square': high_square,
            }, log=self.log_results)

    def add_epoch_metrics(self):
        if self.history.eval_metrics['reward'][-1].data:
            self.history.add_epoch_data({
                'avg_reward': np.mean(self.history.eval_metrics['reward'][-1].data)
            }, log=self.log_results)
        if self.history.eval_metrics['high_square'][-1].data:
            self.history.add_epoch_data({
                'avg_log2_high_square': np.log2(np.mean(self.history.eval_metrics['high_square'][-1].data))
            }, log=self.log_results)
    
    def value_transform(self, value):
        return super().value_transform(value).log()

