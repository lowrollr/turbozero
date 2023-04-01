from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from env import _2048Env, get_legal_actions
import numba

@numba.njit(nogil=True, fastmath=True)
def get_best_move_w_puct(legal_actions, child_n, child_w, child_probs, cpuct):
    n_sum = np.sum(child_n)
    q_values = np.where(child_n != 0, np.divide(child_w, child_n), 0)
    puct_scores = q_values + (cpuct * child_probs * ((np.sqrt(n_sum))/(1 + child_n)))
    legal_move_scores = puct_scores.take(legal_actions)
    best_move = legal_actions[np.argmax(legal_move_scores)]
    return best_move

class PuctNode:
    def __init__(self, move_id, child_probs = None) -> None:
        self.move_id = move_id
        self.n = 0
        self.child_probs = child_probs
        self.children = defaultdict(lambda: [None, None, None, None])
        self.legal_actions = []
        self.cum_child_w = np.zeros(4)
        self.cum_child_n = np.zeros(4)
    
        

class MCTS_Evaluator:
    def __init__(self, model, env, tensor_conversion_fn, cpuct, tau, training=False) -> None:
        self.model = model
        self.env: _2048Env = env
        self.training = training
        self.cpuct = cpuct
        self.tau = tau
        self.tensor_conversion_fn = tensor_conversion_fn
        self.puct_node = PuctNode(None)

    def reset(self):
        self.puct_node = PuctNode(None)
    
    def puct(self, q, n, prior, prev_vists) -> float:
        return q + (self.cpuct * prior * ((np.sqrt(prev_vists))/(1 + n)))
    
    def clear_cache(self):
        self.get_board.cache_clear()

    def iterate(self, move_id: int, puct_node: PuctNode):
        obs, terminated, reward, placement = self.env.push_move(move_id)

        if move_id is not None:
            if placement not in puct_node.children:
                for i in puct_node.legal_actions:
                    puct_node.children[placement][i] = PuctNode(i)
            puct_node = puct_node.children[placement][move_id]
            
        if not terminated:
            if puct_node.n == 0:
                puct_node.legal_actions = np.argwhere(get_legal_actions(self.env.board) == 1).flatten()
                probs, value = self.model(self.tensor_conversion_fn([obs]))
                probs = probs.detach().cpu().numpy()[0]
                reward = value.item()
                puct_node.child_probs = probs
            else:
                best_move = get_best_move_w_puct(puct_node.legal_actions, puct_node.cum_child_n, puct_node.cum_child_w, puct_node.child_probs, self.cpuct)
                reward = self.iterate(best_move, puct_node)
            
                puct_node.cum_child_w[best_move] += reward
                puct_node.cum_child_n[best_move] += 1
        if move_id is not None:
            self.env.moves -=1
        puct_node.n += 1
        return reward
    
    def choose_progression(self, num_iterations=500):
        obs = self.env._get_obs()
            
        original_board = np.copy(self.env.board)
        for _ in range(num_iterations):
            self.iterate(None, self.puct_node)
            self.env.board = np.copy(original_board)

        # pick best (legal) move
        legal_actions = self.puct_node.legal_actions
        
        n_probs = np.copy(self.puct_node.cum_child_n)
        n_probs /= np.sum(n_probs)
        
        n_probs_tau = np.copy(n_probs) ** (1/self.tau)
        n_probs_tau /= np.sum(n_probs_tau)
        
        if self.training:
            selection = legal_actions[np.argmax(np.random.multinomial(1, n_probs_tau.take(legal_actions)))]
        else:
            selection = legal_actions[np.argmax(n_probs_tau.take(legal_actions))]

        _, terminated, reward, placement = self.env.push_move(selection)

        if self.puct_node.children[placement][selection] is None:
            self.puct_node.children[placement][selection] = PuctNode(selection, self.puct_node.child_probs[selection])
        
        self.puct_node = self.puct_node.children[placement][selection]
        if self.puct_node.child_probs is None:
            probs, value = self.model(self.tensor_conversion_fn([obs]))
            probs = probs.detach().cpu().numpy()[0]
            value = value.item()
            self.puct_node.child_probs = probs   
            self.puct_node.n = 1
            self.puct_node.legal_actions = np.argwhere(get_legal_actions(self.env.board)== 1).flatten()

        return terminated, obs, reward, n_probs, selection
    