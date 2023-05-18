from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from env import _2048Env, get_legal_actions
import numba
import torch

@numba.njit(nogil=True, fastmath=True)
def get_best_move_w_puct(legal_actions, child_n, child_w, child_probs, cpuct, lmax, lmin):
    n_sum = np.sum(child_n)

    q_values = np.where(child_n != 0, ((child_w/child_n)-lmin)/(lmax-lmin), np.Inf)
    
    puct_scores = q_values + (cpuct * child_probs * ((np.sqrt(1 + n_sum))/(1 + child_n)))
    legal_move_scores = puct_scores.take(legal_actions)
    # randomly break ties
    best_move = legal_actions[np.random.choice(np.where(legal_move_scores == legal_move_scores.max())[0])]
    return best_move

class PuctNode:
    def __init__(self, move_id, child_probs = None) -> None:
        self.move_id = move_id
        self.n = 0
        self.child_probs = child_probs
        self.children = defaultdict(lambda: dict())
        self.legal_actions = []
        self.cum_child_w = np.zeros(4)
        self.cum_child_n = np.zeros(4)
    
        

class MCTS_Evaluator:
    def __init__(self, model, env, tensor_conversion_fn, cpuct, training=False) -> None:
        self.model = model
        self.env: _2048Env = env
        self.training = training
        self.cpuct = cpuct
        self.lmax = 2
        self.lmin = 1
        self.tensor_conversion_fn = tensor_conversion_fn
        self.puct_node = PuctNode(None)

    def reset(self):
        self.puct_node = PuctNode(None)
    
    def puct(self, q, n, prior, prev_vists) -> float:
        return q + (self.cpuct * prior * ((np.sqrt(prev_vists))/(1 + n)))

    def iterate(self, puct_node: PuctNode):
        if puct_node.n == 0:
            # get legal actions
            puct_node.legal_actions = np.argwhere(get_legal_actions(self.env.board) == 1).flatten()

        # choose which edge to traverse
        best_move = get_best_move_w_puct(puct_node.legal_actions, puct_node.cum_child_n, puct_node.cum_child_w, puct_node.child_probs, self.cpuct, self.lmax, self.lmin)
        if not puct_node.children[best_move]:
            # get all progressions from this move
            boards, progressions, stochastic_probs = self.env.get_progressions_for_move(best_move)
            # convert to tensor
            boards = self.tensor_conversion_fn(boards)
            # get value and probs for each progression
            probs, values = self.model(boards)
            softmax_probs = torch.nn.functional.softmax(probs, dim=-1).detach().cpu().numpy()
            values = values.detach().cpu().numpy()
            self.lmax = max(self.lmax, np.max(values))
            self.lmin = min(self.lmin, np.min(values))

            # create a new node for each progression
            reward = 0
            for i, p_id in enumerate(progressions):
                puct_node.children[best_move][p_id] = PuctNode(best_move, softmax_probs[i])
                reward += stochastic_probs[i] * values[i]
            puct_node.cum_child_n[best_move] = 1
            puct_node.cum_child_w[best_move] = reward
        else:
            # recurse
            _, terminated, _, placement = self.env.push_move(best_move)
            if not terminated:    
                reward = self.iterate(puct_node.children[best_move][placement])
            else:
                reward = 0
            self.env.moves -=1
            puct_node.cum_child_w[best_move] += reward
            puct_node.cum_child_n[best_move] += 1

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
        
        n_probs = np.copy(self.puct_node.cum_child_n)
        n_probs /= np.sum(n_probs)
        
        if self.training:
            selection = legal_actions[np.argmax(np.random.multinomial(1, n_probs.take(legal_actions)))]
        else:
            selection = legal_actions[np.argmax(n_probs.take(legal_actions))]

        deviated = np.argmax(n_probs.take(legal_actions)) != selection

        _, terminated, reward, placement = self.env.push_move(selection, is_simul=False)

        if self.puct_node.children[selection][placement] is None:
            self.puct_node.children[selection][placement] = PuctNode(selection)
        
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
    