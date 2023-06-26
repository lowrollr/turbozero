from copy import deepcopy
from pathlib import Path
from typing import Optional
import torch
from collections import deque
from core import GLOB_FLOAT_TYPE
from core.lz_resnet import LZResnet
from core.history import Metric, TrainingMetrics
from core.hyperparameters import LZHyperparameters
from core.memory import ReplayMemory, ReplayMemory
import numpy as np
import logging
from .lazy_mcts import VectorizedLazyMCTS


class VectTrainer:
    def __init__(self, 
        train_evaluator: VectorizedLazyMCTS,
        test_evaluator: Optional[VectorizedLazyMCTS],
        model: LZResnet, 
        optimizer: torch.optim.Optimizer,
        hypers: LZHyperparameters, 
        num_parallel_envs: int, 
        device: torch.device, 
        history: Optional[TrainingMetrics] = None, 
        memory: Optional[ReplayMemory] = None, 
        log_results: bool = True, 
        interactive: bool = True, 
        run_tag: str = 'model',
    ):
        self.train_evaluator = train_evaluator
        self.train_evaluator.reset()

        self.test_evaluator = test_evaluator
        if self.test_evaluator is not None:
            self.test_evaluator.reset()

        self.log_results = log_results
        self.interactive = interactive
        self.num_parallel_envs = num_parallel_envs

        self.model = model
        self.optimizer = optimizer
        
        self.unfinished_episodes_train = [[] for _ in range(num_parallel_envs)]
        self.unfinished_episodes_test = [[] for _ in range(num_parallel_envs)]

        if memory is None:
            self.memory = ReplayMemory(hypers.replay_memory_size)
        else:
            self.memory = memory

        if history is None:
            self.history = self.init_history()
        else:
            self.history = history
        
        self.hypers = hypers
        self.device = device
        
        self.run_tag = run_tag
        self.already_terminated = set()
    

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

    def init_history(self):
        raise NotImplementedError()

    def push_examples_to_memory_buffer(self, terminated_envs, is_eval, info):
        raise NotImplementedError()
    
    def get_evaluator_and_args(self, is_eval):
        evaluator = self.test_evaluator if is_eval and self.test_evaluator else self.train_evaluator
        iters = self.hypers.num_iters_eval if is_eval else self.hypers.num_iters_train
        depth = self.hypers.iter_depth_test if is_eval else self.hypers.iter_depth_train
        return evaluator, iters, depth

    def run_collection_step(self, is_eval, epsilon=0.0, fixed_batch=False) -> int:
        self.model.eval()
        fused_model = deepcopy(self.model)
        fused_model.fuse()


        evaluator, iters, depth = self.get_evaluator_and_args(is_eval)
        visits = evaluator.explore(fused_model, iters, depth)

        np_states = evaluator.env.states.clone().cpu().numpy()
        np_visits = visits.clone().cpu().numpy()
        
        if torch.rand(1) >= epsilon:
            actions = torch.argmax(visits, dim=1)
        else:
            actions = evaluator.env.fast_weighted_sample(visits)
        terminated, info = evaluator.env.step(actions)
        
        for i in range(evaluator.env.num_parallel_envs):
            if is_eval:
                self.unfinished_episodes_test[i].append((np_states[i], np_visits[i]))
            else:
                self.unfinished_episodes_train[i].append((np_states[i], np_visits[i]))

        terminated_envs = torch.nonzero(terminated).flatten().cpu().tolist()
        num_terminal_envs = len(terminated_envs)

        if num_terminal_envs:
            t_envs = []
            if fixed_batch:
                num_terminal_envs = 0
                for e in terminated_envs:
                    if e not in self.already_terminated:
                        self.already_terminated.add(e)
                        t_envs.append(e)
                        num_terminal_envs += 1
            else:
                t_envs = terminated_envs

            self.add_collection_metrics(t_envs, is_eval)
            self.push_examples_to_memory_buffer(t_envs, is_eval, info)
            if not fixed_batch:
                evaluator.env.reset_invalid_states()

        return num_terminal_envs
    
    def add_collection_metrics(self, terminated_envs, is_eval):
        raise NotImplementedError()
    
    def add_train_metrics(self, policy_loss, value_loss, policy_accuracy, loss):
        raise NotImplementedError()

    def add_epoch_metrics(self):
        raise NotImplementedError()

    def run_training_step(self):
        self.model.train()
        inputs, target_policy, target_value = zip(*self.memory.sample(self.hypers.minibatch_size))
        inputs = torch.from_numpy(np.array(inputs)).to(device=self.device, dtype=GLOB_FLOAT_TYPE)
        target_policy = torch.from_numpy(np.array(target_policy)).to(device=self.device, dtype=GLOB_FLOAT_TYPE)
        target_value = torch.from_numpy(np.array(target_value)).to(device=self.device, dtype=GLOB_FLOAT_TYPE)
        target_policy /= target_policy.sum(dim=1, keepdim=True)

        self.optimizer.zero_grad()
        policy, value = self.model(inputs)
        policy_loss = self.hypers.policy_factor * torch.nn.functional.cross_entropy(policy, target_policy)
        value_loss = torch.nn.functional.mse_loss(value.flatten(), target_value)
        policy_accuracy = torch.eq(torch.argmax(target_policy, dim=1), torch.argmax(policy, dim=1)).to(dtype=GLOB_FLOAT_TYPE).mean()
        loss = policy_loss + value_loss

        loss.backward()
        self.optimizer.step()
        return policy_loss.item(), value_loss.item(), policy_accuracy.item(), loss.item()
    
    def run_test_batch(self) -> None:
        # stores result in eval metrics
        # TODO: maybe return this as its own object instead
        self.already_terminated = set()
        num_terminated = 0
        while num_terminated < self.num_parallel_envs:
            num_terminated += self.run_collection_step(True, fixed_batch=True)
        
    

    def run_training_loop(self, epochs=None) -> None:
        while self.history.cur_epoch < epochs if epochs is not None else True:
            while self.history.cur_train_episode < self.hypers.episodes_per_epoch * (self.history.cur_epoch+1):
                episode_fraction = (self.history.cur_train_episode % self.hypers.episodes_per_epoch) / self.hypers.episodes_per_epoch
                epsilon = max(self.hypers.epsilon_start - (self.hypers.epsilon_decay_per_epoch * (self.history.cur_epoch + episode_fraction)), self.hypers.epsilon_end)
                train_steps = self.run_collection_step(False, epsilon)
                if train_steps:
                    for _ in range(train_steps):
                        cumulative_value_loss = 0.0
                        cumulative_policy_loss = 0.0
                        cumulative_policy_accuracy = 0.0
                        cumulative_loss = 0.0
                        report = True
                        for _ in range(self.hypers.minibatches_per_update):
                            
                            if self.memory.size() > self.hypers.replay_memory_min_size:
                                policy_loss, value_loss, polcy_accuracy, loss = self.run_training_step()
                                cumulative_value_loss += value_loss
                                cumulative_policy_loss += policy_loss
                                cumulative_policy_accuracy += polcy_accuracy
                                cumulative_loss += loss
                            else:
                                logging.info(f'Replay memory size ({self.memory.size()}) <= min size ({self.hypers.replay_memory_min_size}), skipping training step')
                                report = False
                        if report:
                            cumulative_value_loss /= self.hypers.minibatches_per_update
                            cumulative_policy_loss /= self.hypers.minibatches_per_update
                            cumulative_policy_accuracy /= self.hypers.minibatches_per_update
                            cumulative_loss /= self.hypers.minibatches_per_update
                            self.add_train_metrics(cumulative_policy_loss, cumulative_value_loss, cumulative_policy_accuracy, cumulative_loss)
            if self.hypers.eval_episodes_per_epoch > 0 and self.test_evaluator is not None:
                self.test_evaluator.reset()
                self.unfinished_games_test = [[] for _ in range(self.num_parallel_envs)]
                while self.history.cur_test_step < self.hypers.eval_episodes_per_epoch:
                    self.run_collection_step(True)
            
                self.add_epoch_metrics()

            if self.interactive:
                self.history.generate_plots()
            self.save_checkpoint()
            self.history.start_new_epoch()