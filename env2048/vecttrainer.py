from pathlib import Path
import torch
from collections import deque
from az_resnet import AZResnet
from env2048.utils import rotate_training_examples
from env2048.vectenv import Vectorized2048Env
from env2048.vectmcts import Vectorized2048MCTSLazy
from history import Metric, TrainingMetrics
from hyperparameters import AZ_HYPERPARAMETERS, LazyAZHyperparameters
from memory import GameReplayMemory
import numpy as np
import logging

class VectorizedTrainer:
    def __init__(
        self, 
        num_parallel_envs: int, 
        model: AZResnet, 
        optimizer: torch.optim.Optimizer,
        hypers: LazyAZHyperparameters, 
        device: torch.device, 
        history=None, 
        memory=None, 
        log_results=True, 
        interactive=True, 
        run_tag='model'
    ):
        self.log_results = log_results
        self.interactive = interactive
        self.num_parallel_envs = num_parallel_envs
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_evaluator = Vectorized2048MCTSLazy(Vectorized2048Env(num_parallel_envs, device), model, 1)
        self.train_evaluator.reset()
        self.unfinished_games_train = [[] for _ in range(num_parallel_envs)]
        self.unfinished_games_test = [[] for _ in range(num_parallel_envs)]

        if memory is None:
            self.memory = GameReplayMemory(hypers.replay_memory_size)
        else:
            self.memory = memory

        if history is None:
            self.init_history()
        else:
            self.history = history
        
        self.hypers: LazyAZHyperparameters = hypers
        self.device = device

        if hypers.eval_episodes_per_epoch > 0:
            self.test_evaluator = Vectorized2048MCTSLazy(Vectorized2048Env(num_parallel_envs, device), model, 1)
            self.test_evaluator.reset()
        else:
            self.test_evaluator = None
        
        self.run_tag = run_tag
    
    def save_checkpoint(self):
        directory = f'./checkpoints/{self.run_tag}'
        Path(directory).mkdir(parents=True, exist_ok=True)
        filepath = directory + f'/{self.history.cur_epoch}.pt'
        torch.save({
            'parallel_envs': self.num_parallel_envs,
            'model_arch_params': self.model.arch_params,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hypers': self.hypers,
            'history': self.history,
            'memory': self.memory,
            'run_tag': self.run_tag,
            'unfinished_games_train': self.unfinished_games_train,
            'unfinished_games_test': self.unfinished_games_test,
        }, filepath)

    def set_logging_mode(self, on: bool) -> None:
        self.log_results = on
        
    def set_interactive_mode(self, on: bool) -> None:
        self.interactive = on

    def init_history(self):
        self.history = TrainingMetrics(
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

    def push_examples_to_memory_buffer(self, terminated_envs):
        for t in terminated_envs:
            ind = t[0]
            moves = len(self.unfinished_games_train[ind])
            self.assign_remaining_moves(self.unfinished_games_train[ind], moves)
            self.unfinished_games_train[ind] = []

    def assign_remaining_moves(self, states, total_moves):
        game = []
        for i in range(len(states)):
            game.append((*states[i], total_moves - i))
        game = list(rotate_training_examples(game))
        self.memory.insert(game)

    def run_collection_step(self, is_eval, epsilon=0.0):
        evaluator = self.test_evaluator if is_eval and self.test_evaluator else self.train_evaluator
        self.model.eval()
        visits = evaluator.explore(self.hypers.num_iters_train, self.hypers.iter_depth_train)
        np_boards = evaluator.env.boards.clone().cpu().numpy()
        if torch.rand(1) > epsilon:
            actions = torch.argmax(visits, dim=1)
        else:
            actions = torch.multinomial(visits, 1, replacement=True).squeeze(1)
        terminated = evaluator.env.step(actions)
        np_visits = visits.clone().cpu().numpy()
        for i in range(evaluator.env.num_parallel_envs):
            if is_eval:
                self.unfinished_games_test[i].append((np_boards[i], np_visits[i]))
            else:
                self.unfinished_games_train[i].append((np_boards[i], np_visits[i]))
        are_terminal_envs = terminated.any()
        if are_terminal_envs:
            terminated_envs = torch.nonzero(terminated.view(evaluator.env.num_parallel_envs)).cpu().numpy()
            self.add_collection_metrics(terminated_envs, is_eval)
            if not is_eval:
                self.push_examples_to_memory_buffer(terminated_envs)
            evaluator.env.reset_invalid_boards()

        return are_terminal_envs
    
    def add_collection_metrics(self, terminated_envs, is_eval):
        if self.test_evaluator and is_eval:
            high_squares = self.test_evaluator.env.get_high_squares().clone().cpu().numpy()
        else:
            high_squares = self.train_evaluator.env.get_high_squares().clone().cpu().numpy()
        for t in terminated_envs:
            ind = t[0]
            
            if is_eval:
                moves = len(self.unfinished_games_test[ind])
                high_square = high_squares[ind]
                self.history.add_evaluation_data({
                    'reward': moves,
                    'high_square': int(2 ** high_square)
                }, log=self.log_results)
            else:
                moves = len(self.unfinished_games_train[ind])
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
        inputs = torch.from_numpy(np.array(inputs)).to(self.device).float()
        target_policy = torch.from_numpy(np.array(target_policy)).to(self.device).float()
        target_value = torch.from_numpy(np.array(target_value)).to(self.device).float().log()
        target_policy /= target_policy.sum(dim=1, keepdim=True)

        self.optimizer.zero_grad()
        policy, value = self.model(inputs)
        policy_loss = self.hypers.policy_factor * torch.nn.functional.cross_entropy(policy, target_policy)
        value_loss = torch.nn.functional.mse_loss(value.flatten(), target_value)
        policy_accuracy = torch.eq(torch.argmax(target_policy, dim=1), torch.argmax(policy, dim=1)).float().mean()
        loss = policy_loss + value_loss

        loss.backward()
        self.optimizer.step()
        return policy_loss.item(), value_loss.item(), policy_accuracy.item(), loss.item()

    def run_training_loop(self, epochs=None):
        while self.history.cur_epoch < epochs if epochs is not None else True:
            while self.history.cur_train_episode < self.hypers.episodes_per_epoch * (self.history.cur_epoch+1):
                episode_fraction = (self.history.cur_train_episode % self.hypers.episodes_per_epoch) / self.hypers.episodes_per_epoch
                epsilon = max(self.hypers.epsilon_start - (self.hypers.epsilon_decay_per_epoch * (self.history.cur_epoch + episode_fraction)), self.hypers.epsilon_end)
                run_train_step = self.run_collection_step(False, epsilon)
                if run_train_step:
                    cumulative_value_loss = 0.0
                    cumulative_policy_loss = 0.0
                    cumulative_policy_accuracy = 0.0
                    cumulative_loss = 0.0
                    report = True
                    for _ in range(self.hypers.minibatches_per_update):
                        
                        if self.memory.size() > self.hypers.replay_memory_min_size:
                            value_loss, policy_loss, polcy_accuracy, loss = self.run_training_step()
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
            if self.test_evaluator:
                self.test_evaluator.reset()
                self.unfinished_games_test = [[] for _ in range(self.num_parallel_envs)]
                while self.history.cur_test_step < self.hypers.eval_episodes_per_epoch:
                    self.run_collection_step(True)
            
            self.add_epoch_metrics()
            if self.interactive:
                self.history.generate_plots()
            self.history.start_new_epoch()


def load_trainer_from_checkpoint(checkpoint_path, device, load_replay_memory=True):
    checkpoint = torch.load(checkpoint_path)
    hypers = checkpoint['hypers']
    model = AZResnet(checkpoint['model_arch_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hypers.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    history = checkpoint['history']
    history.reset_all_figs()
    memory = checkpoint['memory'] if load_replay_memory else None
    run_tag = checkpoint['run_tag']
    parallel_envs = checkpoint['parallel_envs']
    trainer = VectorizedTrainer(parallel_envs, model, optimizer, hypers, device, history, memory, run_tag=run_tag)
    trainer.unfinished_games_train = checkpoint['unfinished_games_train']
    trainer.unfinished_games_test = checkpoint['unfinished_games_test']
    return trainer