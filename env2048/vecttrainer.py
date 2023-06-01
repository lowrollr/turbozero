import torch
from collections import deque
from history import Metric, TrainingMetrics
from hyperparameters import AZ_HYPERPARAMETERS, LazyAZHyperparameters
from memory import GameReplayMemory
import numpy as np
import logging

class VectorizedTrainer:
    def __init__(self, evaluator, model, hypers, device, history=None, memory=None, log_results=True, interactive=True):
        self.log_results = log_results
        self.interactive = interactive
        self.eval = evaluator
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hypers.learning_rate)
        self.unfinished_games = [[] for _ in range(self.eval.env.num_parallel_envs)]

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
            moves = len(self.unfinished_games[ind])
            self.assign_remaining_moves(self.unfinished_games[ind], moves)
            self.unfinished_games[ind] = []

    def assign_remaining_moves(self, states, total_moves):
        game = []
        for i in range(len(states)):
            game.append((*states[i], total_moves - i))
        self.memory.insert(game)

    def run_collection_step(self, is_eval):
        self.model.eval()
        visits = self.eval.explore(self.hypers.num_iters_train, self.hypers.iter_depth_train)
        np_boards = self.eval.env.boards.clone().cpu().numpy()
        actions = torch.argmax(visits, dim=1)
        terminated = self.eval.env.step(actions)
        np_visits = visits.clone().cpu().numpy()
        for i in range(self.eval.env.num_parallel_envs):
            self.unfinished_games[i].append((np_boards[i], np_visits[i]))
        are_terminal_envs = terminated.any()
        if are_terminal_envs:
            terminated_envs = torch.nonzero(terminated.view(self.eval.env.num_parallel_envs)).cpu().numpy()
            self.add_collection_metrics(terminated_envs, is_eval)
            self.push_examples_to_memory_buffer(terminated_envs)
            self.eval.env.reset_invalid_boards()

        return are_terminal_envs
    
    def add_collection_metrics(self, terminated_envs, is_eval):
        high_squares = self.eval.env.get_high_squares().cpu().numpy()
        for t in terminated_envs:
            ind = t[0]
            moves = len(self.unfinished_games[ind])
            high_square = high_squares[ind]
            if is_eval:
                self.history.add_evaluation_data({
                    'reward': moves,
                    'high_square': int(2 ** high_square)
                }, log=self.log_results)
            else:
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
            self.eval.reset()
            self.unfinished_games = [[] for _ in range(self.eval.env.num_parallel_envs)]
            while self.history.cur_train_episode < self.hypers.episodes_per_epoch * (self.history.cur_epoch+1):
                run_train_step = self.run_collection_step(False)
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

            self.eval.reset()
            self.unfinished_games = [[] for _ in range(self.eval.env.num_parallel_envs)]
            while self.history.cur_test_step < self.hypers.eval_episodes_per_epoch:
                self.run_collection_step(True)
            
            self.add_epoch_metrics()
            if self.interactive:
                self.history.generate_plots()
            self.history.start_new_epoch()



