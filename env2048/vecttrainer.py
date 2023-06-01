import torch
from collections import deque
from hyperparameters import AZ_HYPERPARAMETERS, LazyAZHyperparameters
from memory import GameReplayMemory
import numpy as np

class VectorizedTrainer:
    def __init__(self, evaluator, model, hypers, device):
        self.eval = evaluator
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hypers.learning_rate)
        self.unfinished_games = [[] for _ in range(self.eval.env.num_parallel_envs)]
        self.memory = GameReplayMemory(1000000)
        self.hypers: LazyAZHyperparameters = hypers
        self.device = device


    def push_examples_to_memory_buffer(self, terminated):
        terminated_envs = torch.nonzero(terminated.view(self.eval.env.num_parallel_envs)).cpu().numpy()
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

    def run_collection_step(self):
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
            self.push_examples_to_memory_buffer(terminated)
            self.eval.env.reset_invalid_boards()
        print(np_boards)
        return are_terminal_envs
    
    
    def run_training_step(self):
        self.model.train()
        inputs, target_policy, target_value = zip(*self.memory.sample(self.hypers.minibatch_size))
        inputs = torch.from_numpy(np.array(inputs)).to(self.device).float()
        target_policy = torch.from_numpy(np.array(target_policy)).to(self.device).float()
        target_value = torch.from_numpy(np.array(target_value)).to(self.device).float()
        target_policy /= target_policy.sum(dim=1, keepdim=True)

        self.optimizer.zero_grad()
        policy, value = self.model(inputs)
        policy_loss = self.hypers.policy_factor * torch.nn.functional.cross_entropy(policy, target_policy)
        value_loss = torch.nn.functional.mse_loss(value, target_value)
        policy_accuracy = torch.eq(torch.argmax(target_policy, dim=1), torch.argmax(policy, dim=1)).float().mean()
        loss = policy_loss + value_loss

        loss.backward()
        self.optimizer.step()
        return policy_loss.item(), value_loss.item(), policy_accuracy.item(), loss.item()

    def training_loop(self):
        while True:
            run_train_step = self.run_collection_step()
            if run_train_step:
                cumulative_value_loss = 0.0
                cumulative_policy_loss = 0.0
                cumulative_policy_accuracy = 0.0
                cumulative_loss = 0.0
                for _ in range(self.hypers.minibatches_per_update):
                    
                    if self.memory.size() > self.hypers.replay_memory_min_size:
                        value_loss, policy_loss, polcy_accuracy, loss = self.run_training_step()
                        cumulative_value_loss += value_loss
                        cumulative_policy_loss += policy_loss
                        cumulative_policy_accuracy += polcy_accuracy
                        cumulative_loss += loss
                    else:
                        print('buffer not full')
                    
                    cumulative_value_loss /= self.hypers.minibatches_per_update
                    cumulative_policy_loss /= self.hypers.minibatches_per_update
                    cumulative_policy_accuracy /= self.hypers.minibatches_per_update
                    cumulative_loss /= self.hypers.minibatches_per_update
                    print(cumulative_loss)

