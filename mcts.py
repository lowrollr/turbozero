from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from env import _2048Env

class PuctNode:
    def __init__(self, move_id, prior_prob, child_probs = None, w = 0) -> None:
        self.move_id = move_id
        self.n = 0
        self.w = w
        self.prior_prob = prior_prob
        self.child_probs = child_probs
        self.children = defaultdict(lambda: [None, None, None, None])
        self.legal_actions = []
    
    def update_value(self, new_value):
        self.w += new_value
        self.n += 1
        

class MCTS_Evaluator:
    def __init__(self, model, env, tensor_conversion_fn, training=False, cpuct=2) -> None:
        self.puct_node = None
        self.model = model
        self.env: _2048Env = env
        self.training = training
        self.cpuct = cpuct
        self.tensor_conversion_fn = tensor_conversion_fn

    def reset(self):
        self.puct_node = PuctNode(None, 1.0)
    
    def puct(self, q, n, prior, prev_vists) -> float:
        return q + (self.cpuct * prior * ((np.sqrt(prev_vists))/(1 + n)))
    
    def clear_cache(self):
        self.get_board.cache_clear()

    def initialize_continuations(self, puct_node):
        placements, move_ids, boards, legal_moves = self.env.get_progressions()
        puct_node.legal_actions = legal_moves
        probs, values = self.model(self.tensor_conversion_fn(boards))
        probs = probs.detach().cpu().numpy()
        values = values.detach().cpu().numpy()
        for i,p in enumerate(placements):
            puct_node.children[p][move_ids[i]] = PuctNode(
                move_id=move_ids[i], 
                prior_prob=puct_node.child_probs[move_ids[i]], 
                child_probs=probs[i], 
                w=values[i][0])

    def iterate(self, move_id: int, puct_node: PuctNode):
        obs, terminated, reward, placement = self.env.push_move(move_id)

        if move_id is not None:
            if placement not in puct_node.children:
                for i in puct_node.legal_actions:
                    puct_node.children[placement][i] = PuctNode(i, puct_node.child_probs[i])
            puct_node = puct_node.children[placement][move_id]
            
        if not terminated:
            if puct_node.n == 0:
                puct_node.legal_actions = np.argwhere(self.env.get_legal_actions() == 1).flatten()
                if puct_node.child_probs is None:
                    probs, value = self.model(self.tensor_conversion_fn([obs]))
                    probs = probs.detach().cpu().numpy()[0]
                    reward = value.item()
                    puct_node.child_probs = probs   
                    
                else:
                    # has been pre-initialized, meaning it already has a 'w' value but n == 0, so we really just want to increment n and not change w's value
                    # we can accomplish this by just setting the reward to 0, which will result in no change to w
                    reward = 0
            else:
                legal_actions = puct_node.legal_actions
                cum_w = np.zeros(4, dtype=np.float32)
                cum_n = np.zeros(4, dtype=np.float32)
                for p in puct_node.children:
                    for i in range(4):
                        node = puct_node.children[p][i]
                        if node is not None:
                            cum_w[i] += node.w
                            cum_n[i] += node.n
                q_values = np.zeros(4, dtype=np.float32)
                q_values = np.divide(cum_w, cum_n, where = cum_n > 0)
                puct_scores = np.array([self.puct(q_values[i], cum_n[i], puct_node.child_probs[i], puct_node.n + 1) for i in range(4)])
                legal_move_scores = puct_scores.take(legal_actions)
                best_move = legal_actions[np.argmax(legal_move_scores)]
                reward = self.iterate(best_move, puct_node)
            
        puct_node.update_value(reward)

        return reward
    
    def choose_progression(self, num_iterations=500):
        obs = self.env._get_obs()
        
        if self.puct_node is None: # this is the 'root' state, we need to initialize progressions
            init_probs, init_value = self.model(self.tensor_conversion_fn([obs]))
            init_probs = init_probs.detach().cpu().numpy()[0]
            init_value = init_value.item()
            self.puct_node = PuctNode(
                move_id=None,
                prior_prob=1.0,
                child_probs=init_probs,
                w=init_value)
            self.initialize_continuations(self.puct_node)
        original_board = np.copy(self.env.board)
        for i in range(num_iterations):
            self.iterate(None, self.puct_node)
            self.env.board = np.copy(original_board)


        # pick best (legal) move
        legal_actions = self.puct_node.legal_actions
        n_probs = np.zeros(4)

        for placement_key in self.puct_node.children:
            for i,node in enumerate(self.puct_node.children[placement_key]):
                if node is not None:
                    n_probs[i] += node.n
        
        n_probs /= np.sum(n_probs)
        
        if self.training:
            selection = legal_actions[np.argmax(np.random.multinomial(1, n_probs.take(legal_actions)))]
        else:
            selection = legal_actions[np.argmax(n_probs.take(legal_actions))]

        _, terminated, reward, placement = self.env.push_move(selection)

        if self.puct_node.children[placement][selection] is None:
            self.puct_node.children[placement][selection] = PuctNode(selection, self.puct_node.child_probs[selection])
        
        self.puct_node = self.puct_node.children[placement][selection]
        if self.puct_node.child_probs is None:
            probs, value = self.model(self.tensor_conversion_fn([obs]))
            probs = probs.detach().cpu().numpy()[0]
            value = value.item()
            self.puct_node.child_probs = probs   
            self.puct_node.update_value(value)
            self.puct_node.legal_actions = np.argwhere(self.env.get_legal_actions() == 1).flatten()

        return terminated, obs, reward, n_probs, selection
    