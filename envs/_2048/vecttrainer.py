from pathlib import Path
from typing import Optional
import torch
from collections import deque
from core import GLOB_FLOAT_TYPE
from core.vz_resnet import VZArchitectureParameters, VZResnet
from core.vecttrainer import VectTrainer
from envs._2048.vectenv import _2048Env
from .vectmcts import _2048LazyMCTS
from .utils import rotate_training_examples
from core.history import Metric, TrainingMetrics
from core.hyperparameters import VZHyperparameters
from core.memory import GameReplayMemory, ReplayMemory
import numpy as np
import logging

class _2048Trainer(VectTrainer):
    def __init__(self, 
        train_evaluator: _2048LazyMCTS,
        test_evaluator: Optional[_2048LazyMCTS],
        model: VZResnet, 
        optimizer: torch.optim.Optimizer,
        hypers: VZHyperparameters, 
        num_parallel_envs: int, 
        device: torch.device, 
        history: Optional[TrainingMetrics] = None, 
        memory: Optional[GameReplayMemory] = None, 
        log_results: bool = True, 
        interactive: bool = True, 
        run_tag: Optional[str] = None,
    ):
        run_tag = run_tag or '2048'
        memory = GameReplayMemory(hypers.replay_memory_size)
        super().__init__(
            train_evaluator=train_evaluator,
            test_evaluator=test_evaluator,
            model=model,
            optimizer=optimizer,
            hypers=hypers,
            num_parallel_envs=num_parallel_envs,
            device=device,
            history=history,
            memory=memory,
            log_results=log_results,
            interactive=interactive,
            run_tag=run_tag
        )
        # just to make the linter happy
        self.test_evaluator: _2048LazyMCTS
        self.train_evaluator: _2048LazyMCTS
    


    def init_history(self):
        return TrainingMetrics(
            train_metrics=[
                Metric(name='loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results),
                Metric(name='value_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Value Loss'),
                Metric(name='policy_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Policy Loss'),
                Metric(name='policy_accuracy', xlabel='Step', ylabel='Accuracy (%)', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results, proper_name='Policy Accuracy'),
            ],
            episode_metrics=[
                Metric(name='reward', xlabel='Episode', ylabel='Reward', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results),
                Metric(name='log2_high_square', xlabel='Episode', ylabel='High Tile', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results, proper_name='High Tile (log2)'),
            ],
            eval_metrics=[
                Metric(name='reward', xlabel='Reward', ylabel='Frequency', pl_type='hist',  maximize=True, alert_on_best=False),
                Metric(name='high_square', xlabel='High Tile', ylabel='Frequency Tile', pl_type='bar', maximize=True, alert_on_best=False, proper_name='High Tile'),
            ],
            epoch_metrics=[
                Metric(name='avg_reward', xlabel='Epoch', ylabel='Average Reward', maximize=True, alert_on_best=self.log_results, proper_name='Average Reward'),
                Metric(name='avg_log2_high_square', xlabel='Epoch', ylabel='Average High Tile', maximize=True, alert_on_best=self.log_results, proper_name='Average High Tile (log2)'),
            ]
        )

    def push_examples_to_memory_buffer(self, terminated_envs, is_eval):
        for t in terminated_envs:
            ind = t[0]
            if not is_eval:
                moves = len(self.unfinished_episodes_train[ind])
                self.assign_remaining_moves(self.unfinished_episodes_train[ind], moves)
                self.unfinished_episodes_train[ind] = []
            else:
                self.unfinished_episodes_test[ind] = []

    def assign_remaining_moves(self, states, total_moves):
        game = []
        for i in range(len(states)):
            game.append((*states[i], total_moves - i))
        game = list(rotate_training_examples(game))
        self.memory.insert(game)
    
    def add_collection_metrics(self, terminated_envs, is_eval):
        if self.test_evaluator and is_eval:
            high_squares = self.test_evaluator.env.get_high_squares().clone().cpu().numpy()
        else:
            high_squares = self.train_evaluator.env.get_high_squares().clone().cpu().numpy()
        for t in terminated_envs:
            ind = t[0]
            
            if is_eval:
                moves = len(self.unfinished_episodes_test[ind])
                high_square = high_squares[ind]
                self.history.add_evaluation_data({
                    'reward': moves,
                    'high_square': int(2 ** high_square)
                }, log=self.log_results)
            else:
                moves = len(self.unfinished_episodes_train[ind])
                high_square = high_squares[ind]
                self.history.add_episode_data({
                    'reward': moves,
                    'log2_high_square': int(high_square)
                }, log=self.log_results)
    
    def add_train_metrics(self, policy_loss, value_loss, policy_accuracy, loss):
        self.history.add_training_data({
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_accuracy': policy_accuracy,
            'loss': loss
        }, log=self.log_results)

    def add_epoch_metrics(self):
        self.history.add_epoch_data({
            'avg_reward': np.mean(self.history.eval_metrics['reward'][-1].data),
            'avg_log2_high_square': np.log2(np.mean(self.history.eval_metrics['high_square'][-1].data))
        }, log=self.log_results)

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

    
def init_new_2048_trainer(
        arch_params: VZArchitectureParameters,
        parallel_envs: int,
        device: torch.device,
        hypers: VZHyperparameters,
        log_results=True,
        interactive=True,
        run_tag: Optional[str] = None
    ) -> _2048Trainer:
        model = VZResnet(arch_params).to(device)
        if GLOB_FLOAT_TYPE == torch.float16:
            model = model.half()
        optimizer = torch.optim.AdamW(model.parameters(), lr=hypers.learning_rate)

        train_evaluator = _2048LazyMCTS(_2048Env(parallel_envs, device), model, hypers.mcts_c_puct)
        test_evaluator = _2048LazyMCTS(_2048Env(parallel_envs, device), model, hypers.mcts_c_puct)

        return _2048Trainer(train_evaluator, test_evaluator, model, optimizer, hypers, parallel_envs, device, None, None, log_results, interactive, run_tag)

def init_2048_trainer_from_checkpoint(
        parallel_envs: int, 
        checkpoint_path: str, 
        device: torch.device, 
        memory: Optional[GameReplayMemory] = None, 
        log_results=True, 
        interactive=True
    ) -> _2048Trainer:
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hypers: VZHyperparameters = checkpoint['hypers']
    model = VZResnet(checkpoint['model_arch_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if GLOB_FLOAT_TYPE == torch.float16:
        model = model.half()

    optimizer = torch.optim.AdamW(model.parameters(), lr=hypers.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    history = checkpoint['history']
    history.reset_all_figs()
    run_tag = checkpoint['run_tag']

    train_evaluator = _2048LazyMCTS(_2048Env(parallel_envs, device), hypers.mcts_c_puct)
    test_evaluator = _2048LazyMCTS(_2048Env(parallel_envs, device), hypers.mcts_c_puct)
    trainer = _2048Trainer(train_evaluator, test_evaluator, model, optimizer, hypers, parallel_envs, device, history, memory, log_results, interactive, run_tag=run_tag)
    return trainer