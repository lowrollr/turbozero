import random
from collections import deque, OrderedDict
import numpy as np
import torch
from dataclasses import dataclass
from env import _2048Env
from mcts import MCTS_Evaluator
import time
import bottleneck
import os 
import numba

class ReplayMemory:
    def __init__(self, max_size=10000) -> None:
        self.max_size = max_size
        self.memory = deque([], maxlen=max_size)
        pass

    def sample(self, num_samples): 
        return random.sample(self.memory, num_samples)
    
    def insert(self, sample):
        self.memory.append(sample)

    def size(self):
        return len(self.memory)
    
class GameReplayMemory(ReplayMemory):
    def __init__(self, max_size=10000) -> None:
        super().__init__(max_size)
    
    def sample(self, num_samples):
        games = random.choices(self.memory, k=num_samples)
        samples = []
        for game in games:
            samples.append(random.sample(game, 1)[0])
        return samples


@dataclass()
class MCTS_HYPERPARAMETERS:
    lr: float = 5e-4
    weight_decay: float = 1e-3
    minibatch_size: int = 1000
    replay_memory_size: int = 30000 
    num_mcts_train_evals: int = 500
    num_mcts_test_evals: int = 1000
    num_epochs: int = 1000
    num_eval_games: int = 100
    episodes_per_epoch: int = 1000
    num_episodes: int = 1000
    checkpoint_every: int = 100
    mcts_c_puct: int = 12
    mcts_epsilon_start: float = 1.0
    mcts_epsilon_end: float = 0.01
    mcts_epsilon_decay_rate: float = 0.0001
    minibatches_per_episode: int = 5
    c_prob: int = 3

import matplotlib.pyplot as plt
import IPython.display as display

class MetricsHistory:
    def __init__(self) -> None:
        self.cur_epoch = 0
        self.training_history = OrderedDict()
        self.training_history['game_score'] = []
        self.training_history['game_moves'] = []
        self.training_history['high_square'] = []
        self.training_history['prob_loss'] = []
        self.training_history['value_loss'] = []
        self.training_history['total_loss'] = []
        
        
        self.eval_history = []

        self.eval_history_overall = OrderedDict()
        self.eval_history_overall['game_score'] = []
        self.eval_history_overall['game_moves'] = []
        self.eval_history_overall['high_square'] = []


        self.episodes = 0
        self.best_training_result = float('-inf')
        self.best_eval_result = float('-inf')
        self.training_figs = [plt.figure() for _ in range(6)]
        self.eval_figs = [plt.figure() for _ in range(3)]
        self.high_score_metric = 'game_moves'
        self.plot_titles = ['Game Score', 'Game Moves', 'High Square (log2)', 'Prob Loss', 'Value Loss', 'Total Loss']

        self.last_eval_histograms = []
    
    def add_training_history(self, info):
        
        self.training_history['game_score'].append(info['game_score'])
        self.training_history['game_moves'].append(info['game_moves'])
        self.training_history['prob_loss'].append(info['prob_loss'])
        self.training_history['value_loss'].append(info['value_loss'])
        self.training_history['total_loss'].append(info['total_loss'])
        self.training_history['high_square'].append(info['high_square'])
        self.episodes += 1
        if info[self.high_score_metric] > self.best_training_result:
            self.best_training_result = info[self.high_score_metric]
            return True
        return False
    
    def add_eval_history(self, info):
        if len(self.eval_history) < self.cur_epoch + 1:
            self.eval_history.append(OrderedDict())
            self.eval_history[self.cur_epoch]['game_score'] = []
            self.eval_history[self.cur_epoch]['game_moves'] = []
            self.eval_history[self.cur_epoch]['high_square'] = []

        epoch_history = self.eval_history[self.cur_epoch]
        epoch_history['game_score'].append(info['game_score'])
        epoch_history['game_moves'].append(info['game_moves'])
        epoch_history['high_square'].append(info['high_square'])

        if info[self.high_score_metric] > self.best_eval_result:
            self.best_eval_result = info[self.high_score_metric]
            return True
        return False
    
    def update_overall_eval_history(self):
        for i, (k, data) in enumerate(self.eval_history[self.cur_epoch].items()):
            mean = np.mean(data)
            self.eval_history_overall[k].append(mean)
            fig = self.eval_figs[i]
            fig.clear()
            ax = fig.add_subplot(111)
            ax.plot(self.eval_history_overall[k])
            ax.set_title(f'Mean Eval {self.plot_titles[i]}')
            ax.annotate('%0.3f' % mean, xy=(1, mean), xytext=(8, 0), 
                                    xycoords=('axes fraction', 'data'), textcoords='offset points')
    def clear_eval_history(self):
        self.eval_history = []
        self.eval_history_overall = OrderedDict()
        self.eval_history_overall['game_score'] = []
        self.eval_history_overall['game_moves'] = []
        self.eval_history_overall['high_square'] = []
        self.cur_epoch = 0
        self.best_eval_result = float('-inf')
        self.eval_figs = [plt.figure() for _ in range(3)]

    def set_last_eval_plots(self):
        figs = []
        for i, data in enumerate(self.eval_history[self.cur_epoch].values()):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(data, bins='auto')
            ax.set_title(f'Epoch {self.cur_epoch} {self.plot_titles[i]}')
            figs.append(fig)
        self.last_eval_histograms = figs

    def increment_epoch(self):
        self.cur_epoch += 1

    def get_move_mean_and_stddev(self):
        return np.mean(self.eval_history[-1]['game_moves']), np.std(self.eval_history[-1]['game_moves'])

    def plot_history(self, offset=0, window_size=50, plot_training = True, plot_eval = True):
        # plot training history
        qty = len(self.training_history['game_score'])
        offset = min(offset, qty)
        offset = -offset
        
        window_size = min(window_size, qty)
        display.clear_output(wait=False)
        if plot_training:
            for i, data in enumerate(self.training_history.values()):
                fig = self.training_figs[i]
                fig.clear()
                ax = fig.add_subplot(111)
                ts = np.arange(self.episodes+offset+1, self.episodes+1)
                running_mean = bottleneck.move_mean(data[offset:], window=window_size, min_count=1)
                ax.plot(ts, data[offset:])
                ax.plot(ts, running_mean)
                ax.set_title(self.plot_titles[i])
                ax.annotate('%0.3f' % running_mean[-1], xy=(1, running_mean[-1]), xytext=(8, 0), 
                                    xycoords=('axes fraction', 'data'), textcoords='offset points')
                display.display(fig)
        if plot_eval:
            for fig in self.last_eval_histograms:
                display.display(fig)

            # plot last eval results
            for fig in self.eval_figs:
                display.display(fig)

        

        
        

def load_from_checkpoint(filename, model_class, load_replay_memory=True):
    run_tag = filename.split('_')[0]
    checkpoint = torch.load(filename)
    hyperparameters = checkpoint['hyperparameters']
    episode = checkpoint['episode']
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.share_memory()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.lr, weight_decay=hyperparameters.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    metrics_history = checkpoint['metrics_history']
    memory = None
    if load_replay_memory:
        memory = checkpoint.get('replay_memory')
    elif memory is None:
        memory = ReplayMemory(hyperparameters.replay_memory_size)
    
    return episode, model, optimizer, hyperparameters, metrics_history, memory, run_tag
    

def save_checkpoint(episodes, model, optimizer, hyperparameters, metrics_history, replay_memory, run_tag='', save_replay_memory=True):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': hyperparameters,
        'metrics_history': metrics_history,
        'replay_memory': replay_memory if save_replay_memory else None,
        'episode': episodes,
        'model_type': str(type(model))
    }, f'{run_tag}_ep{episodes}.pt')



MOVE_MAP = {0: 'right', 1: 'up', 2: 'left', 3: 'down'}
def test_network(model, hyperparameters, tensor_conversion_fn, debug_print=False):
    env = _2048Env()
    mcts = MCTS_Evaluator(model, env, tensor_conversion_fn, cpuct=hyperparameters.mcts_c_puct, training=False)
    env.reset()
    model.eval()
    with torch.no_grad():
        moves = 0
        while True:
            start_time = time.time()
            probs, value = model(tensor_conversion_fn([env.board]))
            if debug_print:
                print(env.board)
            terminated, _, reward, mcts_probs, move = mcts.choose_progression(hyperparameters.num_mcts_test_evals)
            moves += 1
            if debug_print:
                print(f'Time elapsed: {time.time() - start_time}s')
                print(f'Move #: {moves}')
                print(f'Move: {MOVE_MAP[move]}')
                print(f'Network Probs: {probs.detach().numpy()}')
                print(f'MCTS Probs: {mcts_probs}')
                print(f'Network value: {value.item()}')
                print(f'Q Value: {np.sum(mcts.puct_node.cum_child_w) / mcts.puct_node.n}')
            if terminated:
                if debug_print:
                    print(f'Terminated, final reward = {reward}')
                break
    return reward, moves, env.get_highest_square(), env.get_score()

def train(samples, model, optimizer, tensor_conversion_fn, c_prob=5):
    model.train()
    obs, mcts_probs, rewards = zip(*samples)
    obs = tensor_conversion_fn(obs)
    mcts_probs = torch.from_numpy(np.array(mcts_probs))
    rewards = torch.from_numpy(np.array(rewards)).unsqueeze(1).float()
    optimizer.zero_grad()

    exp_probs, exp_rewards = model(obs)
    value_loss = torch.mean(torch.abs(rewards - exp_rewards))
    prob_loss = c_prob * torch.nn.functional.cross_entropy(exp_probs, mcts_probs)
    # prob_loss = -1 * c_prob * torch.mean(torch.sum(mcts_probs*torch.log(exp_probs), dim=1))

    loss = value_loss + prob_loss
    loss.backward()
    optimizer.step()
    return value_loss.item(), prob_loss.item(), loss.item()

def collect_episode(model, hyperparameters, tensor_conversion_fn, tau_shift, tau_divisor):
    model.eval()
    training_examples = []
    env = _2048Env()
    env.reset()
    mcts = MCTS_Evaluator(model, env, tensor_conversion_fn=tensor_conversion_fn, cpuct=hyperparameters.mcts_c_puct, tau_shift=tau_shift, tau_divisor=tau_divisor, training=True)
    moves = 0
    with torch.no_grad():
        while True:
            # get inputs, reward, mcts probs, run n_iterations of MCTS
            terminated, inputs, reward, mcts_probs, _ = mcts.choose_progression(hyperparameters.num_mcts_train_evals)
            moves += 1
            training_examples.append([inputs, mcts_probs])
            if terminated:
                break
        rem_reward = reward
        for example in training_examples:
            example.append(rem_reward)
            rem_reward -= 1

    return training_examples, reward, moves, env.get_highest_square(), os.getpid(), env.get_score()


def rotate_training_examples(training_examples):
    inputs, probs, rewards = zip(*training_examples)
    rotated_inputs = []
    for i in inputs:
        for k in range(4):
            rotated_inputs.append(np.rot90(i, k=k))
    rotated_probs = []
    for p in probs:
        # left -> down
        # down -> right
        # right -> up
        # up -> left
        for k in range(4):
            rotated_probs.append(np.roll(p, k))
    rotated_rewards = []
    for _ in range(4):
        rotated_rewards.extend(rewards)
    
    return zip(rotated_inputs, rotated_probs, rotated_rewards)