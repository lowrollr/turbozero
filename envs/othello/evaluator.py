

from typing import Optional, Union
import torch
from core.evaluation.lazy_mcts import VectorizedLazyMCTS
from core.evaluation.lazy_mcts_hypers import LazyMCTSHypers
from core.evaluation.mcts import VectorizedMCTS
from core.evaluation.mcts_hypers import MCTSHypers
from envs.othello.vectenv import OthelloVectEnv


class OthelloLazyMCTS(VectorizedLazyMCTS):
    def __init__(self, num_parallel_envs: int, device: torch.device, board_size: int, hypers: LazyMCTSHypers, env: Optional[OthelloVectEnv] = None, debug: bool = False) -> None:
        env = OthelloVectEnv(num_parallel_envs, device, board_size=board_size, debug=debug) if env is None else env
        super().__init__(env, hypers)

class OthelloMCTS(VectorizedMCTS):
    def __init__(self, num_parallel_envs: int, device: torch.device, board_size: int, hypers: MCTSHypers, env: Optional[OthelloVectEnv] = None, debug: bool = False) -> None:
        env = OthelloVectEnv(num_parallel_envs, device, board_size=board_size, debug=debug) if env is None else env
        super().__init__(env, hypers)

OTHELLO_EVALUATORS = Union[OthelloLazyMCTS, OthelloMCTS]