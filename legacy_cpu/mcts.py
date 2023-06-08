from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from legacy_cpu.env import _2048Env, get_legal_actions
import numba
import torch

from legacy_cpu.stochastic_sp_env import SpStochasticMCTSEnv



# We use a modified version of the PUCT algorithm from the AlphaZero paper to choose which move to explore next.
# The first term of PUCT (q-value) represents a bias towards exploitation of moves with a higher predicted score
# The second term (UCB) represents a bias towards exploration of moves with a higher prior probability, and introduces additional bias towards exploration of moves that have been explored less.
# I've made an adjustment where q-values are normalized to the range [0, 1] before being used in the PUCT calculation.
@numba.njit(nogil=True, fastmath=True)
def get_best_action(legal_actions, child_n, child_w, child_probs, cpuct):
    n_sum = np.sum(child_n)
    w_sum = np.sum(child_w)

    # we want unvisited nodes to have a very high, non-infinite values (so that relative order according to probabilities is preserved)
    q_values = np.where(child_n != 0, (child_w/child_n), w_sum * 100000000000000)
    
    puct_scores = q_values + (cpuct * child_probs * ((np.sqrt(1 + n_sum))/(1 + child_n)))
    legal_action_scores = puct_scores * legal_actions

    best_action = np.argmax(legal_action_scores)
    return best_action


# a single MCTS node
class GameStateNode:
    def __init__(self, move_id, child_probs = None) -> None:
        self.move_id = move_id # digit from 0-3 representing the move that led to this node
        self.n = 0 # number of times this node has been visited
        self.child_probs = child_probs # prior probabilities for each child node
        self.children = defaultdict(lambda: dict()) # map from move_id to child node
        self.legal_actions = np.ndarray([]) # list of legal actions from this node
        self.pior_w = np.zeros(4) # accumulated scores for each child node
        self.pior_n = np.zeros(4) # number of times each child node has been visited
        

class MCTS_Evaluator:
    def __init__(self, model, env, tensor_conversion_fn, cpuct, epsilon=1.0, training=False) -> None:
        self.model = model
        self.env: SpStochasticMCTSEnv = env
        self.training = training
        self.cpuct = cpuct
        self.tensor_conversion_fn = tensor_conversion_fn
        self.puct_node = GameStateNode(None)
        self.epsilon = epsilon
    
    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.puct_node = GameStateNode(None)

    def iterate(self, puct_node: GameStateNode):
        if puct_node.n == 0:
            # get legal actions
            puct_node.legal_actions = self.env.get_legal_actions()

        # choose which edge to traverse
        best_action = get_best_action(puct_node.legal_actions, puct_node.pior_n, puct_node.pior_w, puct_node.child_probs, self.cpuct)
        if not puct_node.children[best_action]:
            # get all progressions from this move
            boards, progressions, stochastic_probs = self.env.get_progressions(best_action)
            # convert to tensor
            boards = self.tensor_conversion_fn(boards)
            # get value and probs for each progression
            probs, values = self.model(boards)
            softmax_probs = torch.nn.functional.softmax(probs, dim=-1).detach().cpu().numpy()
            values = values.detach().cpu().numpy()

            # create a new node for each progression
            reward = 0
            for i, (p_id, term) in enumerate(progressions):
                puct_node.children[best_action][p_id] = GameStateNode(best_action, softmax_probs[i])
                if not term:
                    reward += stochastic_probs[i] * values[i]
            puct_node.pior_n[best_action] = 1
            puct_node.pior_w[best_action] = reward
        else:
            # recurse
            _, _, terminated, _, info = self.env.step(best_action)
            if not terminated: 
                placement = info['progression_id']   
                reward = self.iterate(puct_node.children[best_action][placement])
            else:
                reward = 0
            puct_node.pior_w[best_action] += reward
            puct_node.pior_n[best_action] += 1

        puct_node.n += 1
        return reward
    
    def choose_progression(self, num_iterations):
        # if this node is not already visited, get the child probabilities from the model
        initial_observation = self.env._get_obs()
        if self.puct_node.child_probs is None:
            probs, _ = self.model(self.tensor_conversion_fn([initial_observation]))
            probs = torch.nn.functional.softmax(probs, dim=-1).detach().cpu().numpy()[0]
            self.puct_node.child_probs = probs

    
        original_env = deepcopy(self.env)

        for _ in range(num_iterations):
            self.iterate(self.puct_node)
            self.env = deepcopy(original_env)

        mcts_probs = np.copy(self.puct_node.pior_n)
        mcts_probs /= np.sum(mcts_probs)
        
        if self.training and self.env.np_random.random() < self.epsilon:
            
            best_action = np.argmax(self.env.np_random.multinomial(1, mcts_probs))
        else:
            best_action = np.argmax(mcts_probs)

        _, reward, terminated, _, info = self.env.step(best_action)
        placement = info['progression_id']

        if self.puct_node.children[best_action][placement] is None:
            self.puct_node.children[best_action][placement] = GameStateNode(best_action)
        
        self.puct_node = self.puct_node.children[best_action][placement]

        # if self.puct_node.child_probs is None:
        #     # initialize empty node
        #     probs, value = self.model(self.tensor_conversion_fn([obs]))
        #     probs = torch.nn.functional.softmax(probs, dim=-1).detach().cpu().numpy()[0]
        #     value = value.item()
        #     self.puct_node.child_probs = probs   
        #     self.puct_node.n = 1
        #     self.puct_node.legal_actions = self.env.get_legal_actions()

        return terminated, initial_observation, reward, mcts_probs, info
    