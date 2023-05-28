import gymnasium as gym
import numpy as np

from gymnasium import spaces

from stochastic_sp_env import SpStochasticMCTSEnv
from .utils import apply_move, post_move, get_legal_actions, get_progressions_for_board

from copy import deepcopy

class Env2048(SpStochasticMCTSEnv):
    def __init__(self, size=4) -> None:
        self.size = 4
        
        self.observation_space = spaces.Tuple(
            [spaces.Tuple([spaces.Discrete(18) for _ in range(self.size)]) for _ in range(self.size)]
        )

        # 0 -> right
        # 1 -> up
        # 2 -> left
        # 3 -> down
        self.action_space = spaces.Discrete(4)

        self.board = np.zeros((self.size, self.size), dtype=np.int32)

        self.score = 0
        self.moves = 0
        self.high_square = 0

    def _get_obs(self):
        return tuple(tuple(row) for row in self.board)    
    
    def _get_info(self):
        return {'score': self.score, 'moves': self.moves, 'high_square': self.get_high_square(), 'progression_id': None}
    
    def get_high_square(self):
        return 2 ** np.max(self.board)

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        self.moves = 0
        self.score = 0
        num_starting_tiles = options.get('starting_tiles', 2)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        # choose 2 random indices for the first 2 tiles
        indices = self.np_random.choice(self.size ** 2, num_starting_tiles, replace=False)
        for index in indices:
            self.board[index//self.size, index%self.size] = 1
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        score = apply_move(self.board, action)
        self.score += score
        self.moves += 1
        placement, terminated = post_move(self.board, self.np_random)

        info = {
            'score': self.score,
            'moves': self.moves,
            'max_tile': self.get_high_square(),
            'progression_id': placement
        }

        return self._get_obs(), 0, terminated, False, info
    
    def get_progressions(self, action) -> np.ndarray:
        board = deepcopy(self.board)
        apply_move(board, action)
        return get_progressions_for_board(board)
    
    def get_legal_actions(self):
        return get_legal_actions(self.board)