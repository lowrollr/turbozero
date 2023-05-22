from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from env import _2048Env, get_legal_actions
import numba
import torch


# We use a modified version of the PUCT algorithm from the AlphaZero paper to choose which move to explore next.
# The first term of PUCT (q-value) represents a bias towards exploitation of moves with a higher predicted score
# The second term (UCB) represents a bias towards exploration of moves with a higher prior probability, and introduces additional bias towards exploration of moves that have been explored less.
# I've made an adjustment where q-values are normalized to the range [0, 1] before being used in the PUCT calculation.
@numba.njit(nogil=True, fastmath=True)
def get_best_move_w_puct(legal_actions, child_n, child_w, child_probs, cpuct):
    n_sum = np.sum(child_n)

    q_values = np.where(child_n != 0, (child_w/child_n), np.Inf)
    
    puct_scores = q_values + (cpuct * child_probs * ((np.sqrt(1 + n_sum))/(1 + child_n)))
    legal_move_scores = puct_scores.take(legal_actions)
    # randomly break ties
    best_move = legal_actions[np.random.choice(np.where(legal_move_scores == legal_move_scores.max())[0])]
    return best_move


# a single MCTS node
class GameStateNode:
    def __init__(self, move_id, child_probs = None) -> None:
        self.move_id = move_id # digit from 0-3 representing the move that led to this node
        self.n = 0 # number of times this node has been visited
        self.child_probs = child_probs # prior probabilities for each child node
        self.children = defaultdict(lambda: dict()) # map from move_id to child node
        self.legal_actions = [] # list of legal actions from this node
        self.pior_w = np.zeros(4) # accumulated scores for each child node
        self.pior_n = np.zeros(4) # number of times each child node has been visited
        

class MCTS_Evaluator:
    def __init__(self, model, env, tensor_conversion_fn, cpuct, exploration_cutoff=None, epsilon=None, training=False) -> None:
        self.model = model
        self.env: _2048Env = env
        self.training = training
        self.cpuct = cpuct
        self.tensor_conversion_fn = tensor_conversion_fn
        self.puct_node = GameStateNode(None)
        self.epsilon = epsilon
        self.exploration_cutoff = exploration_cutoff
    
    def reset(self):
        self.puct_node = GameStateNode(None)
    
    def puct(self, q, n, prior, prev_vists) -> float:
        return q + (self.cpuct * prior * ((np.sqrt(prev_vists))/(1 + n)))

    def iterate(self, puct_node: GameStateNode):
        if puct_node.n == 0:
            # get legal actions
            puct_node.legal_actions = np.argwhere(get_legal_actions(self.env.board) == 1).flatten()

        # choose which edge to traverse
        best_move = get_best_move_w_puct(puct_node.legal_actions, puct_node.pior_n, puct_node.pior_w, puct_node.child_probs, self.cpuct)
        if not puct_node.children[best_move]:
            # get all progressions from this move
            boards, progressions, stochastic_probs = self.env.get_progressions_for_move(best_move)
            # convert to tensor
            boards = self.tensor_conversion_fn(boards)
            # get value and probs for each progression
            probs, values = self.model(boards)
            softmax_probs = torch.nn.functional.softmax(probs, dim=-1).detach().cpu().numpy()
            values = values.detach().cpu().numpy()

            # create a new node for each progression
            reward = 0
            for i, (p_id, term) in enumerate(progressions):
                puct_node.children[best_move][p_id] = GameStateNode(best_move, softmax_probs[i])
                if not term:
                    reward += stochastic_probs[i] * values[i]
            puct_node.pior_n[best_move] = 1
            puct_node.pior_w[best_move] = reward
        else:
            # recurse
            _, terminated, _, placement = self.env.push_move(best_move)
            if not terminated:    
                reward = self.iterate(puct_node.children[best_move][placement])
            else:
                reward = 0
            self.env.moves -=1
            puct_node.pior_w[best_move] += reward
            puct_node.pior_n[best_move] += 1

        puct_node.n += 1
        return reward


    
    def choose_progression(self,num_iterations=500):
        obs = self.env._get_obs()
            
        original_board = np.copy(self.env.board)

        probs, value = self.model(self.tensor_conversion_fn([obs]))
        probs = torch.nn.functional.softmax(probs, dim=-1).detach().cpu().numpy()[0]
        self.lmax = value.item()
        self.lmin = self.lmax - 1
        self.puct_node.child_probs = probs


        for _ in range(num_iterations):
            self.iterate(self.puct_node)
            self.env.board = np.copy(original_board)

        # pick best (legal) move
        legal_actions = self.puct_node.legal_actions
        
        n_probs = np.copy(self.puct_node.pior_n)
        n_probs /= np.sum(n_probs)
        
        if self.training and \
            (self.exploration_cutoff is None or self.env.moves < self.exploration_cutoff) and \
            (self.epsilon is None or np.random.random() < self.epsilon):
            
            selection = legal_actions[np.argmax(np.random.multinomial(1, n_probs.take(legal_actions)))]
        else:
            selection = legal_actions[np.argmax(n_probs.take(legal_actions))]

        deviated = np.argmax(n_probs.take(legal_actions)) != selection

        _, terminated, reward, placement = self.env.push_move(selection, is_simul=False)

        if self.puct_node.children[selection][placement] is None:
            self.puct_node.children[selection][placement] = GameStateNode(selection)
        
        self.puct_node = self.puct_node.children[selection][placement]

        if self.puct_node.child_probs is None:
            # initialize empty node
            probs, value = self.model(self.tensor_conversion_fn([obs]))
            probs = torch.nn.functional.softmax(probs, dim=-1).detach().cpu().numpy()[0]
            value = value.item()
            self.puct_node.child_probs = probs   
            self.puct_node.n = 1
            self.puct_node.legal_actions = np.argwhere(get_legal_actions(self.env.board)== 1).flatten()

        return terminated, obs, reward, n_probs, selection, deviated
    