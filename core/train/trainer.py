from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

from tqdm import tqdm
import torch
import logging
from pathlib import Path
from core.algorithms.evaluator import EvaluatorConfig
from core.resnet import TurboZeroResnet
from core.test.tester import TesterConfig, Tester
from core.train.collector import Collector
from core.utils.history import Metric, TrainingMetrics
from collections import deque

from core.utils.memory import GameReplayMemory, ReplayMemory
import time


def init_history(log_results: bool = True):
    return TrainingMetrics(
        train_metrics=[
            Metric(name='loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 25}, maximize=False, alert_on_best=log_results),
            Metric(name='value_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 25}, maximize=False, alert_on_best=log_results, proper_name='Value Loss'),
            Metric(name='policy_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 25}, maximize=False, alert_on_best=log_results, proper_name='Policy Loss'),
            Metric(name='policy_accuracy', xlabel='Step', ylabel='Accuracy (%)', addons={'running_mean': 25}, maximize=True, alert_on_best=log_results, proper_name='Policy Accuracy'),
            Metric(name='replay_memory_similarity', xlabel='Step', ylabel='Cosine Centroid Similarity', addons={'running_mean': 25}, maximize=False, alert_on_best=log_results, proper_name='Replay Memory Similarity'),
        ],
        episode_metrics=[],
        eval_metrics=[],
        epoch_metrics=[]
    )


@dataclass
class TrainerConfig:
    algo_config: EvaluatorConfig
    episodes_per_epoch: int
    episodes_per_minibatch: int
    minibatch_size: int
    learning_rate: float
    momentum: float
    c_reg: float
    lr_decay_gamma: float    
    parallel_envs: int
    policy_factor: float
    replay_memory_min_size: int
    replay_memory_max_size: int
    test_config: TesterConfig
    replay_memory_sample_games: bool = True


class Trainer:
    def __init__(self,
        config: TrainerConfig,
        collector: Collector,
        tester: Tester,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        raw_train_config: dict,
        raw_env_config: dict,
        history: TrainingMetrics,
        log_results: bool = True,
        interactive: bool = True,
        run_tag: str = 'model',
        debug: bool = False
    ):
        self.collector = collector
        self.tester = tester
        self.parallel_envs = collector.evaluator.env.parallel_envs
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.log_results = log_results
        self.interactive = interactive
        self.run_tag = run_tag
        self.raw_train_config = raw_train_config
        self.raw_env_config = raw_env_config
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.lr_decay_gamma)
        self.debug = debug
        self.collection_queue = deque()

        self.history = history 

        if config.replay_memory_sample_games:
            self.replay_memory = GameReplayMemory(
                config.replay_memory_max_size
            )
        else:
            self.replay_memory = ReplayMemory(
                config.replay_memory_max_size,
            )
    
    def add_collection_metrics(self, episodes):
        raise NotImplementedError()
    
    def add_train_metrics(self, policy_loss, value_loss, policy_accuracy, loss):
        self.history.add_training_data({
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_accuracy': policy_accuracy,
            'loss': loss
        }, log=self.log_results)

    def add_epoch_metrics(self):
        raise NotImplementedError()
    
    def training_step(self):
        inputs, target_policy, target_value, legal_actions = zip(*self.replay_memory.sample(self.config.minibatch_size))
        inputs = torch.stack(inputs).to(device=self.device)

        target_policy = torch.stack(target_policy).to(device=self.device)
        target_policy = self.policy_transform(target_policy)

        target_value = torch.stack(target_value).to(device=self.device)
        target_value = self.value_transform(target_value)

        legal_actions = torch.stack(legal_actions).to(device=self.device)

        self.optimizer.zero_grad()
        policy_logits, values = self.model(inputs)
        # multiply policy logits by legal actions mask, set illegal actions to smallest possible negative float32
        # consistent with open_spiel implementation
        policy_logits = (policy_logits * legal_actions) + (torch.finfo(torch.float32).min * (~legal_actions))

        policy_loss = self.config.policy_factor * torch.nn.functional.cross_entropy(policy_logits, target_policy)
        # multiply by 2 since most other implementations have values rangeing from -1 to 1 whereas ours range from 0 to 1
        # this makes values loss a bit more comparable
        value_loss = torch.nn.functional.mse_loss(values.flatten() * 2, target_value * 2)
        loss = policy_loss + value_loss

        policy_accuracy = torch.eq(torch.argmax(target_policy, dim=1), torch.argmax(policy_logits, dim=1)).float().mean()

        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), policy_accuracy.item(), loss.item()
    
    def policy_transform(self, policy):
        return policy
    
    def value_transform(self, value):
        return value
    
    def train_minibatch(self):
        self.model.train()
        memory_size = self.replay_memory.size()
        if memory_size >= self.config.replay_memory_min_size:
            policy_loss, value_loss, polcy_accuracy, loss = self.training_step()
            self.add_train_metrics(policy_loss, value_loss, polcy_accuracy, loss)
        else:
            logging.info(f'Replay memory samples ({memory_size}) <= min samples ({self.config.replay_memory_min_size}), skipping training step')
        
    def train_epoch(self):
        new_episodes = 0
        threshold_for_training_step = self.config.episodes_per_minibatch
        if not self.debug:
            progress_bar = tqdm(total=self.config.episodes_per_epoch, desc='Collecting self-play episodes...', leave=True, position=0)
        while new_episodes < self.config.episodes_per_epoch:
            if self.collection_queue:
                episode = self.collection_queue.popleft()
                self.replay_memory.insert(episode)
                if not self.debug:
                    progress_bar.update(1)
                new_episodes += 1
                if new_episodes >= threshold_for_training_step:
                    threshold_for_training_step += self.config.episodes_per_minibatch
                    self.train_minibatch()
            else:
                finished_episodes, _ = self.collector.collect()
                if finished_episodes:
                    for episode in finished_episodes:
                        episode = self.collector.postprocess(episode)
                        if new_episodes >= self.config.episodes_per_epoch:
                            self.collection_queue.append(episode)
                        else:
                            self.replay_memory.insert(episode)
                            if not self.debug:
                                progress_bar.update(1)
                            new_episodes += 1
                            if new_episodes >= threshold_for_training_step:
                                threshold_for_training_step += self.config.episodes_per_minibatch
                                self.train_minibatch()
                    
                self.add_collection_metrics(finished_episodes)
    
    def fill_replay_memory(self):
        if not self.debug:
            progress_bar = tqdm(total=self.config.replay_memory_min_size, desc='Populating Replay Memory...', leave=True, position=0)
        while self.replay_memory.size() < self.config.replay_memory_min_size:
            finished_episodes, _ = self.collector.collect()
            if finished_episodes:
                for episode in finished_episodes:
                    episode = self.collector.postprocess(episode)
                    self.replay_memory.insert(episode)
                    if not self.debug:
                        progress_bar.update(1)
        

    def training_loop(self, epochs: Optional[int] = None):
        total_epochs = self.history.cur_epoch + epochs if epochs is not None else None
        if self.history.cur_epoch == 0:
            # run initial test batch with untrained model
            if self.tester.config.episodes_per_epoch > 0:
                self.tester.collect_test_batch()
            self.save_checkpoint()
            
        if self.replay_memory.size() <= self.config.replay_memory_min_size:
            logging.info('Populating replay memory...')
            self.fill_replay_memory()

        while self.history.cur_epoch < total_epochs if total_epochs is not None else True:
            self.history.start_new_epoch()
            self.train_epoch()

            if self.tester.config.episodes_per_epoch > 0:
                self.tester.collect_test_batch()
            self.add_epoch_metrics()

            if self.interactive:
                self.history.generate_plots()

            self.scheduler.step()
            self.save_checkpoint()
            
            
            
    def save_checkpoint(self, custom_name: Optional[str] = None) -> None:
        directory = f'./checkpoints/{self.run_tag}/'
        Path(directory).mkdir(parents=True, exist_ok=True)
        filename = custom_name if custom_name is not None else str(self.history.cur_epoch)
        filepath = directory + f'{filename}.pt'
        torch.save({
            'model_arch_params': self.model.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'run_tag': self.run_tag,
            'raw_train_config': self.raw_train_config,
            'raw_env_config': self.raw_env_config
        }, filepath)

    def benchmark_collection_step(self):
        start = time.time()
        self.collector.collect()
        tottime = time.time() - start
        time_per_env_step = tottime / self.config.parallel_envs
        print(f'Stepped {self.config.parallel_envs} envs in {tottime:.4f} seconds ({time_per_env_step:.4f} seconds per step)')


