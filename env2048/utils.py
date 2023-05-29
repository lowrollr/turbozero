from typing import List, Tuple
from numba import njit
import numpy as np

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
    score = 0

    while index >= 0 and index < size:
        if values[index] != 0:
            if can_combine and merged[m_index] == values[index]:
                merged[m_index] += 1
                score += 2 ** merged[m_index]
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
    return merged, score


def apply_move(board, action):
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
    score = 0

    size = board.shape[0]
    if is_rows:
        for i in range(size):
            board[i], s = merge(board[i], reverse=reverse)
            score += s
    else:
        for i in range(size):
            board[:, i], s = merge(board[:, i], reverse=reverse)
            score += s
    return score

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
def get_progressions_for_board(board) -> Tuple[List[np.ndarray], List[Tuple[Tuple[int, int], bool]], List[float]]:
        boards = []
        progressions = []
        probs = []
        # returns board, probability tuples for each possible progression
        empty_squares = np.argwhere(board == 0)
        num_empty_squares = len(empty_squares)
        fraction = 1 / num_empty_squares
        for (i0, i1) in empty_squares:
            new_board1 = np.copy(board)
            new_board1[i0,i1] = 1
            boards.append(new_board1)
            probs.append(fraction * 0.9)
            new_board2 = np.copy(board)
            new_board2[i0,i1] = 2
            boards.append(new_board2)
            probs.append(fraction * 0.1)
            p_id = (i0 * 4) + i1
            progressions.append(((p_id, 1), get_legal_actions(new_board1).sum() == 0))
            progressions.append(((p_id, 2), get_legal_actions(new_board1).sum() == 0))
        return boards, progressions, probs

def post_move(board, rand):
    terminated = False
    placement = None
    # choose a random empty spot
    empty = np.argwhere(board == 0)
    index = rand.choice(empty.shape[0], 1)[0]
    value = 2 if rand.random() >= 0.9 else 1
    board[empty[index, 0], empty[index, 1]] = value
    placement = ((empty[index, 0] * 4) + empty[index, 1], value)

    if np.max(get_legal_actions(board)) == 0:
        terminated = True
    
    return placement, terminated

def compare_tiles(arr):
    vertical_comparison = np.pad(np.logical_and(np.equal(arr[:-1, :], arr[1:, :]), np.not_equal(arr[:-1, :], 0)).astype(float), ((0, 1), (0, 0)), mode='constant', constant_values=0)
    horizontal_comparison = np.pad(np.logical_and(np.equal(arr[:, :-1], arr[:, 1:]), np.not_equal(arr[:, :-1], 0)).astype(float), ((0, 0), (0, 1)), mode='constant', constant_values=0)

    return vertical_comparison, horizontal_comparison

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
    for r in rewards:
        rotated_rewards.extend([r] * 4)
    
    return zip(rotated_inputs, rotated_probs, rotated_rewards)