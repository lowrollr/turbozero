
from types import FunctionType
from typing import Iterable
import numpy as np
import torch

from az_resnet import AZResnet
from .env import Env2048

from history import TrainingMetrics, Metric
from mcts import MCTS_Evaluator
from .utils import compare_tiles, rotate_training_examples


from trainer import AlphaZeroTrainer


class _2048Trainer(AlphaZeroTrainer):
    def convert_obs_batch_to_tensor(self, obs_batch: Iterable[np.ndarray]) -> torch.Tensor:
        tensors = []
        for board in obs_batch:
            board = np.array(board)
            tensor = torch.stack([
                torch.from_numpy(np.equal(board, 0)).to(self.device).float(),
                *torch.from_numpy(np.stack(compare_tiles(board), axis=0)).to(self.device).float(),
                torch.from_numpy(board).to(self.device).float()
            ], dim=0)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)
    
    def init_history(self) -> TrainingMetrics:
        return TrainingMetrics(
            train_metrics= [
                Metric(name='loss', xlabel='Episode', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results),
                Metric(name='value_loss', xlabel='Episode', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Value Loss'),
                Metric(name='policy_loss', xlabel='Episode', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Policy Loss'),
                Metric(name='policy_accuracy', xlabel='Episode', ylabel='Accuracy (%)', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results, proper_name='Policy Accuracy'),
                Metric(name='score', xlabel='Episode', ylabel='Score', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results),
                Metric(name='moves', xlabel='Episode', ylabel='Moves', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results),
                Metric(name='max_tile', xlabel='Episode', ylabel='Max Tile', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results),
            ],
            eval_metrics= [
                Metric(name='max_tile', xlabel='Max Tile', ylabel='Frequency', pl_type='hist', maximize=True, alert_on_best=False),
                Metric(name='score', xlabel='Score', ylabel='Frequency', pl_type='hist', maximize=True, alert_on_best=False),
                Metric(name='moves', xlabel='Moves', ylabel='Frequency', pl_type='hist', maximize=True, alert_on_best=False),
            ],
            epoch_metrics= [
                Metric(name='avg_score', xlabel='Epoch', ylabel='Average Reward', maximize=True, alert_on_best=self.log_results),
                Metric(name='avg_moves', xlabel='Epoch', ylabel='Average Moves', maximize=True, alert_on_best=self.log_results),
                Metric(name='avg_max_tile', xlabel='Epoch', ylabel='Average Max Tile', maximize=True, alert_on_best=self.log_results)
            ]
        )
    
    def add_training_episode_to_history(self, value_loss, policy_loss, loss, policy_accuracy, reward, info) -> None:
        self.history.add_training_data({
            'loss': loss,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'policy_accuracy': policy_accuracy,
            'score': info['score'],
            'moves': info['moves'],
            'max_tile': info['max_tile']
        })

    def add_eval_episode_to_history(self, reward, info) -> None:
        self.history.add_evaluation_data({
            'score': info['score'],
            'moves': info['moves'],
            'max_tile': info['max_tile']
        })

    def add_eval_data_to_epoch_history(self) -> None:
        self.history.add_epoch_data({
            'avg_score': np.mean(self.history.eval_metrics['score'][-1].data),
            'avg_moves': np.mean(self.history.eval_metrics['moves'][-1].data),
            'avg_max_tile': np.mean(self.history.eval_metrics['max_tile'][-1].data)
        }, log=self.log_results)

    @staticmethod
    def initialize_evaluator(model: AZResnet, tensor_conversion_fn, cpuct: float, epsilon: float, training: bool) -> MCTS_Evaluator:
        env = Env2048()
        return MCTS_Evaluator(
            model=model,
            env=env,
            tensor_conversion_fn=tensor_conversion_fn,
            cpuct=cpuct,
            epsilon=epsilon,
            training=training
        )

    def enque_training_samples(self, training_examples):
        training_examples = rotate_training_examples(training_examples)
        return super().enque_training_samples(training_examples)
    
