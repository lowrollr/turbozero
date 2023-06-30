from typing import Optional
import torch
import logging
from pathlib import Path
from core.collector import Collector
from core.history import Metric, TrainingMetrics

from core.hyperparameters import LZHyperparameters
from core.memory import ReplayMemory


class Trainer:
    def __init__(self,
        num_parallel_envs: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        hypers: LZHyperparameters,
        device: torch.device,
        episode_memory_device: torch.device = torch.device('cpu'),
        history: Optional[TrainingMetrics] = None,
        log_results: bool = True,
        interactive: bool = True,
        run_tag: str = 'model',
    ):
        self.model = model
        self.optimizer = optimizer
        self.hypers = hypers
        self.device = device
        self.episode_memory_device = episode_memory_device
        self.log_results = log_results
        self.interactive = interactive
        self.run_tag = run_tag

        self.history = history if history else self.init_history()

        self.train_collector: Collector
        self.test_collector: Collector
        self.replay_memory: ReplayMemory


    def init_history(self) -> TrainingMetrics:
        return TrainingMetrics(
            train_metrics=[
                Metric(name='loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results),
                Metric(name='value_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Value Loss'),
                Metric(name='policy_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Policy Loss'),
                Metric(name='policy_accuracy', xlabel='Step', ylabel='Accuracy (%)', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results, proper_name='Policy Accuracy'),
            ],
            episode_metrics=[],
            eval_metrics=[],
            epoch_metrics=[]
        )
    
    def add_collection_metrics(self, episodes):
        raise NotImplementedError()
    
    def add_evaluation_metrics(self, episodes):
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
    
    def training_steps(self, num_steps: int):
        if num_steps > 0:
            self.model.train()
            memory_size = self.replay_memory.size()
            if memory_size >= self.hypers.replay_memory_min_size:
                for _ in range(num_steps):
                    minibatch_value_loss = 0.0
                    minibatch_policy_loss = 0.0
                    minibatch_policy_accuracy = 0.0
                    minibatch_loss = 0.0
                    
                    for _ in range(self.hypers.minibatches_per_update):
                        policy_loss, value_loss, polcy_accuracy, loss = self.training_step()
                        minibatch_value_loss += value_loss
                        minibatch_policy_loss += policy_loss
                        minibatch_policy_accuracy += polcy_accuracy
                        minibatch_loss += loss

                    minibatch_value_loss /= self.hypers.minibatches_per_update
                    minibatch_policy_loss /= self.hypers.minibatches_per_update
                    minibatch_policy_accuracy /= self.hypers.minibatches_per_update
                    minibatch_loss /= self.hypers.minibatches_per_update
                    self.add_train_metrics(minibatch_policy_loss, minibatch_value_loss, minibatch_policy_accuracy, minibatch_loss)
            else:
                logging.info(f'Replay memory samples ({memory_size}) <= min samples ({self.hypers.replay_memory_min_size}), skipping training steps')
    
    def training_step(self):
        inputs, target_policy, target_value = zip(*self.replay_memory.sample(self.hypers.minibatch_size))

        inputs = torch.stack(inputs).to(device=self.device)
        target_policy = torch.stack(target_policy).to(device=self.device)
        target_policy.div_(target_policy.sum(dim=1, keepdim=True))
        target_value = torch.stack(target_value).to(device=self.device)

        self.optimizer.zero_grad()
        policy, value = self.model(inputs)

        policy_loss = self.hypers.policy_factor * torch.nn.functional.cross_entropy(policy, target_policy)
        value_loss = torch.nn.functional.mse_loss(value.flatten(), target_value)
        loss = policy_loss + value_loss

        policy_accuracy = torch.eq(torch.argmax(target_policy, dim=1), torch.argmax(policy, dim=1)).float().mean()

        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), policy_accuracy.item(), loss.item()
    
    def selfplay_step(self):
        episode_fraction = (self.history.cur_train_episode % self.hypers.episodes_per_epoch) / self.hypers.episodes_per_epoch
        epsilon = max(self.hypers.epsilon_start - (self.hypers.epsilon_decay_per_epoch * (self.history.cur_epoch + episode_fraction)), self.hypers.epsilon_end)
        finished_episodes, _ = self.train_collector.collect(self.model, epsilon=epsilon)
        if finished_episodes:
            for episode in finished_episodes:
                episode = self.train_collector.postprocess(episode)
                self.replay_memory.insert(episode)
                self.add_collection_metrics(episode)
        
        num_train_steps = len(finished_episodes)
        self.training_steps(num_train_steps)

    def evaluation_step(self):
        episodes, termianted = self.test_collector.collect(self.model, epsilon=0.0)
        self.add_evaluation_metrics(episodes)
        return termianted
    
    def evaluate_n_episodes(self, num_episodes: int):
        self.test_collector.reset()
        # we make a big assumption here that all evaluation episodes can be run in parallel
        # if the user wants to evaluate an obscene number of episodes this will become problematic
        # TODO: batch evaluation episodes where necessary
        completed_episodes = torch.zeros(num_episodes, dtype=torch.bool)
        while not completed_episodes.all():
            termianted = self.evaluation_step()
            completed_episodes &= termianted

    def training_loop(self, epochs: Optional[int] = None):
        while self.history.cur_epoch < epochs if epochs is not None else True:
            while self.history.cur_train_step < self.hypers.episodes_per_epoch * (self.history.cur_epoch+1):
                self.selfplay_step()
            
            self.evaluate_n_episodes(self.hypers.eval_episodes_per_epoch)
            self.add_epoch_metrics()

            if self.interactive:
                self.history.generate_plots()
            self.save_checkpoint()
            self.history.start_new_epoch()

    def save_checkpoint(self, custom_name: Optional[str] = None) -> None:
        directory = f'./checkpoints/{self.run_tag}/'
        Path(directory).mkdir(parents=True, exist_ok=True)
        filename = custom_name if custom_name is not None else str(self.history.cur_epoch)
        filepath = directory + f'{filename}.pt'
        torch.save({
            'model_arch_params': self.model.arch_params,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hypers': self.hypers,
            'history': self.history,
            'run_tag': self.run_tag
        }, filepath)