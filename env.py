import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from numba import njit

@njit(nogil=True, fastmath=True)
def merge(values, reverse=False):
    index = 0
    direction = 1
    size = len(values)
    if reverse:
        index = size - 1
        direction = -1
    merged = np.zeros(4)
    seen_first = False
    can_combine = False
    m_index = index
    while index >= 0 and index < size:
        if values[index] != 0:
            if can_combine and merged[m_index] == values[index]:
                merged[m_index] += 1
                can_combine = False
            elif values[index] != 0:
                if seen_first:
                    m_index += direction
                    merged[m_index] = values[index]
                    can_combine = True
                else:
                    merged[m_index] = values[index]
                    can_combine = True
                    seen_first = True
                
        index += direction
    return merged

@njit(nogil=True, fastmath=True)
def get_legal_actions(board):
    legal_actions = np.zeros(4)
    for p0 in range(3):
        if np.any(np.logical_and(board[p0] == board[p0+1], board[p0] != 0)):
            legal_actions[1] = 1
            legal_actions[3] = 1
            break
        if np.any(np.logical_and(board[p0] == 0, board[p0+1] != 0)):
            legal_actions[1] = 1
            if legal_actions[3]:
                break
        if np.any(np.logical_and(board[p0+1] == 0, board[p0] != 0)):
            legal_actions[3] = 1
            if legal_actions[1]:
                break

    for p0 in range(3):
        if np.any(np.logical_and(board[:, p0] == board[:, p0+1], board[:, p0] != 0)):
            legal_actions[0] = 1
            legal_actions[2] = 1
            break
        if np.any(np.logical_and(board[:, p0] == 0, board[:, p0+1] != 0)):
            legal_actions[2] = 1
            if legal_actions[0]:
                break
        if np.any(np.logical_and(board[:, p0+1] == 0, board[:, p0] != 0)):
            legal_actions[0] = 1
            if legal_actions[2]:
                break
    return legal_actions

@njit(nogil=True, fastmath=True)
def post_move(board):
    terminated = False
    placement = None
    # choose a random empty spot
    empty = np.argwhere(board == 0)
    index = np.random.choice(empty.shape[0], 1)[0]
    value = 2 if random.random() >= 0.9 else 1
    board[empty[index, 0], empty[index, 1]] = value
    placement = ((empty[index, 0] * 4) + empty[index, 1], value)

    if np.max(get_legal_actions(board)) == 0:
        terminated = True
    
    return placement, terminated

class _2048Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4):
        self.size = size
        self.window_size = 512
        
        self.observation_space = spaces.Tuple(
            [spaces.Tuple([spaces.Discrete(18) for _ in range(self.size)]) for _ in range(self.size)]
        )

        # We can slide in any of the four directions: right, up, left, down.
        self.action_space = spaces.Discrete(4) # 0 == right, 1 == up, 2 == left, 3 == down
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.moves = 0
    
    def _get_obs(self):
        return tuple(tuple(row) for row in self.board)
    
    def _get_info(self):
        return None

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        self.moves = 0
        num_starting_tiles = options.get('starting_tiles', 2)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        # choose 2 random indices for the first 2 tiles
        indices = np.random.choice(self.size ** 2, num_starting_tiles, replace=False)
        for index in indices:
            self.board[index//self.size, index%self.size] = 1
        return self._get_obs(), self._get_info()
        
    def apply_move(self, board, action):
        # execute action
        reverse = True
        is_rows = True
        if action == 1:
            reverse = False
            is_rows = False
        elif action == 2:
            reverse = False
            is_rows = True
        elif action == 3:
            reverse = True
            is_rows = False

        if is_rows:
            for i in range(self.size):
                board[i] = merge(board[i], reverse=reverse)
        else:
            for i in range(self.size):
                board[:, i] = merge(board[:, i], reverse=reverse)

    def step(self, action):
        self.apply_move(self.board, action)
    
        placement, terminated = post_move(self.board)
        if terminated:
            reward = self.moves
        else:
            reward = 0
            
        return self._get_obs(), reward, terminated, False, self._get_info(), placement
    
    def get_progressions(self):
        legal_moves = np.argwhere(get_legal_actions(self.board) == 1).flatten()
        boards = []
        move_ids = []
        placements = []

        for move in legal_moves:
            new_board = np.copy(self.board)
            self.apply_move(new_board, move)
            empty_squares = np.argwhere(new_board == 0)
            for a in range(1, 3):
                for (i0, i1) in empty_squares:
                    newer_board = np.copy(new_board)
                    newer_board[i0,i1] = a
                    boards.append(newer_board)
                    move_ids.append(move)
                    placements.append(((i0 * 4) + i1, a))


        return placements, move_ids, boards, legal_moves
    
    def push_move(self, move_id):
        if move_id is not None:
            self.moves += 1
            obs, reward, terminated, _, _, placement = self.step(move_id)
            return obs, terminated, reward, placement
        else:
            return self._get_obs(), False, 0, None
    
    def get_highest_square(self):
        return np.max(self.board)

    def render(self):
        return None
    
    def close(self):
        return None
    