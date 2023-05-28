
from types import FunctionType
from typing import Callable, Iterable, List, Optional, Tuple
import torch
torch.set_num_threads(1)
import numpy as np
from history import TrainingMetrics
from hyperparameters import AZ_HYPERPARAMETERS
from mcts import MCTS_Evaluator
from memory import GameReplayMemory
from pathlib import Path
from az_resnet import AZResnet
from history import TrainingMetrics, Metric
import logging
import torch.multiprocessing as mp
from utils import apply_async_dill

class AlphaZeroTrainer:
    def __init__(self, model, optimizer, device, hypers, history=None, memory=None, run_tag='', log_results=True):
        self.device = device
        self.model: AZResnet = model
        self.model.share_memory()
        self.optimizer: torch.optim.Optimizer = optimizer
        self.hypers: AZ_HYPERPARAMETERS = hypers
        self.run_tag: str = run_tag
        self.log_results: bool = log_results
        self.interactive = False
        self.plot_every = 25
        self.num_processes = mp.cpu_count()
        self.epsilon = 1.0
       
        if history is None:
            self.init_history()
        else:
            self.history: TrainingMetrics = history

        if memory is None:
            self.init_memory(hypers.replay_memory_size)
        else:
            self.memory: GameReplayMemory = memory
        
        

    def set_interactive(self, interactive: bool):
        self.interactive = interactive
    
    def set_plot_every(self, plot_every: int):
        self.plot_every = plot_every

    def convert_obs_batch_to_tensor(self, obs_batch: Iterable[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.array(obs_batch)).to(self.device).float()
    
    def init_memory(self, size):
        self.memory = GameReplayMemory(size)

    def init_history(self) -> None:
        self.history = TrainingMetrics(
            train_metrics = [
                Metric(name='loss', xlabel='Episode', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results),
                Metric(name='value_loss', xlabel='Episode', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Value Loss'),
                Metric(name='policy_loss', xlabel='Episode', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Policy Loss'),
                Metric(name='policy_accuracy', xlabel='Episode', ylabel='Accuracy (%)', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results, proper_name='Policy Accuracy'),
                Metric(name='reward', xlabel='Episode', ylabel='Reward', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results),
            ],
            eval_metrics = [
                Metric(name='reward', xlabel='Reward', ylabel='Frequency', pl_type='hist', maximize=True, alert_on_best=False),
            ],
            epoch_metrics = [
                Metric(name='avg_reward', xlabel='Epoch', ylabel='Average Reward', maximize=True, alert_on_best=self.log_results),
            ]
        )
    
    def add_training_episode_to_history(self, value_loss, policy_loss, loss, policy_accuracy, reward, info) -> None:
        self.history.add_training_data({
            'loss': loss,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'policy_accuracy': policy_accuracy,
            'reward': reward,
        }, log=self.log_results)

    def add_evaluation_episode_to_history(self, reward, info) -> None:
        self.history.add_evaluation_data({
            'reward': reward,
        }, log=self.log_results)

    def add_eval_data_to_epoch_history(self) -> None:
        self.history.add_epoch_data({
            'avg_reward': np.mean(self.history.eval_metrics['reward'][-1].data),
        }, log=self.log_results)

    def train_batch(self, observations: Iterable[np.ndarray], mcts_probs: Iterable[np.ndarray], rewards: Iterable[float]) \
                                                                                -> Tuple[float, float, float, float]:
        self.model.train()
        
        obs = self.convert_obs_batch_to_tensor(observations)
        probs = torch.from_numpy(np.array(mcts_probs)).float()
        rew = torch.from_numpy(np.array(rewards)).float()

        self.optimizer.zero_grad(set_to_none=True)

        exp_probs, exp_rewards = self.model(obs)

        policy_loss = self.hypers.policy_factor * torch.nn.functional.cross_entropy(exp_probs, probs)
        value_loss = torch.nn.functional.mse_loss(exp_rewards, rew)

        policy_accuracy = torch.eq(torch.argmax(exp_probs, dim=1), torch.argmax(probs, dim=1)).float().mean()

        loss = policy_loss + value_loss

        loss.backward()
        self.optimizer.step()

        return value_loss.item(), policy_loss.item(), loss.item(), policy_accuracy.item()
    
    @staticmethod
    def collect_test_episode(model: AZResnet, evaluator: MCTS_Evaluator, iters: int) -> Tuple[float, dict]:
        model.eval()
        evaluator.reset()
        with torch.no_grad():
            while True:
                terminated, _, reward, _, info = evaluator.choose_progression(iters)
                if terminated:
                    break
        return reward, info
    
    @staticmethod
    def collect_training_episode(model: AZResnet, evaluator: MCTS_Evaluator, iters: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], float, dict]:
        model.eval()
        evaluator.reset()
        training_examples = []
        while True:
            terminated, obs, reward, probs, info = evaluator.choose_progression(iters)
            training_examples.append((obs, probs, reward))
            if terminated:
                break

        return training_examples, reward, info
    
    @staticmethod
    def initialize_evaluator(model: AZResnet, tensor_conversion_fn, cpuct: float, epsilon: float, training: bool) -> MCTS_Evaluator:
        # env = TODO
        # return MCTS_Evaluator(model, env, tensor_conversion_fn, cpuct, epsilon, training)
        raise NotImplementedError('initialize_evaluator must be implemented by subclass') 
    
    def save_checkpoint(self, save_replay_memory=True) -> None:
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'arch_params': self.model.arch_params,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'hypers': self.hypers,
            'memory': self.memory if save_replay_memory else None,
            'run_tag': self.run_tag
        }
        full_path = Path(f'checkpoints/{self.run_tag}')
        full_path.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, f'{full_path}/{self.history.cur_epoch}.pt')
    
    def enque_training_samples(self, training_examples):
        self.memory.insert(list(training_examples))
    
        
    def train_step(self) -> Optional[Tuple[float, float, float, float]]:
        if self.memory.size() >= self.hypers.replay_memory_min_size:
            cum_vl, cum_pl, cum_tl, cum_pa = 0.0, 0.0, 0.0, 0.0
            for _ in range(self.hypers.minibatches_per_update):
                obs, probs, rewards = zip(*self.memory.sample(self.hypers.minibatch_size))
                value_loss, prob_loss, total_loss, policy_acc = self.train_batch(obs, probs, rewards)
                cum_vl += value_loss
                cum_pl += prob_loss
                cum_tl += total_loss
                cum_pa += policy_acc
            cum_vl /= self.hypers.minibatches_per_update
            cum_pl /= self.hypers.minibatches_per_update
            cum_tl /= self.hypers.minibatches_per_update
            cum_pa /= self.hypers.minibatches_per_update

            return cum_vl, cum_pl, cum_tl, cum_pa
        else:
            logging.info(f'Replay memory size not large enough, {self.memory.size()} < {self.hypers.replay_memory_min_size}')
    
    def run_training_loop(self) -> None:
        # we need to wrap class methods in a function to pass to mp.Pool
        def collect_training_episode_wrapper():
            evaluator = self.initialize_evaluator(self.model, self.convert_obs_batch_to_tensor, self.hypers.mcts_c_puct, self.epsilon, training=True)
            return self.collect_training_episode(self.model, evaluator, self.hypers.mcts_iters_train)
        
        def test_wrapper():
            evaluator = self.initialize_evaluator(self.model, self.convert_obs_batch_to_tensor, self.hypers.mcts_c_puct, self.epsilon, training=False)
            return self.collect_test_episode(self.model, evaluator, self.hypers.mcts_iters_eval)
        
        def enque_and_train(training_samples):
            training_examples, reward, info = training_samples
            self.enque_training_samples(training_examples)
            results = self.train_step()
            if results is not None:
                value_loss, policy_loss, total_loss, policy_accuracy = results
                self.add_training_episode_to_history(value_loss, policy_loss, total_loss, policy_accuracy, reward, info)
                if self.interactive:
                    if self.history.cur_epoch % self.plot_every == 0:
                        self.history.generate_plots()
        
        def add_evaluation_episode_to_history_wrapper(results):
            self.add_evaluation_episode_to_history(*results)

        # make sure replay memory is filled up to min size
        logging.info('Pre-populating replay memory...')
        with mp.Pool(self.num_processes) as pool:
            results = []
            for _ in range(self.memory.size(), self.hypers.replay_memory_min_size):
                results.append(
                    apply_async_dill(
                        pool,
                        collect_training_episode_wrapper,
                        (),
                        callback=enque_and_train,
                        error_callback=print
                    )
                )
            for r in results:
                r.wait()
        
        while self.history.cur_epoch < self.hypers.num_epochs:
            cur_epoch = self.history.cur_epoch
            logging.info(f'Starting epoch {cur_epoch}/{self.hypers.num_epochs}')
            logging.info('Training...')

            if self.hypers.epsilon_decay_per_epoch:
                self.epsilon = np.clip(1.0 - (cur_epoch * self.hypers.epsilon_decay_per_epoch), 0.0, 1.0)

            with mp.Pool(self.num_processes) as pool:
                results = []
                for _ in range(self.history.cur_train_episode, self.hypers.episodes_per_epoch * cur_epoch):
                    results.append(
                        apply_async_dill(
                            pool,
                            collect_training_episode_wrapper,
                            (),
                            callback=enque_and_train,
                            error_callback=print
                        )
                    )
                for r in results:
                    r.wait()

            if self.hypers.eval_games > 0:
                logging.info('Testing...')
                with mp.Pool(self.num_processes) as pool:
                    results = []
                    for _ in range(self.hypers.eval_games):
                        results.append(
                            apply_async_dill(
                                pool,
                                test_wrapper,
                                (),
                                callback=add_evaluation_episode_to_history_wrapper,
                                error_callback=print
                            )
                        )
                    for r in results:
                        r.wait()

                self.add_eval_data_to_epoch_history()
                if self.interactive:
                    self.history.generate_plots()

                self.history.start_new_epoch()
            logging.info('Saving checkpoint...')
            self.save_checkpoint()
            logging.info('Saved checkpoint!')
            


def init_trainer_from_checkpoint(filename, load_replay_memory=True):
    checkpoint = torch.load(filename)
    
    hypers = checkpoint['hypers']

    model = AZResnet(checkpoint['arch_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=hypers.learning_rate, weight_decay=hypers.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    history = checkpoint['history']
    history.reset_all_figs()

    memory = checkpoint['memory'] if load_replay_memory else None

    run_tag = checkpoint['run_tag']

    return AlphaZeroTrainer(model, optimizer, hypers, history, memory, run_tag)





