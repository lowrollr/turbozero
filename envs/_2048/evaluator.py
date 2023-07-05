
from typing import Optional, Union
from core.evaluation.lazy_mcts import VectorizedLazyMCTS
from core.evaluation.lazy_mcts_hypers import LazyMCTSHypers
from core.evaluation.mcts import VectorizedMCTS
from core.evaluation.mcts_hypers import MCTSHypers
from envs._2048.vectenv import _2048Env




class _2048LazyMCTS(VectorizedLazyMCTS):
    def __init__(self, num_parallel_envs, device, hypers: LazyMCTSHypers, env: Optional[_2048Env] = None, debug: bool = False) -> None:
        env = _2048Env(num_parallel_envs, device, debug) if env is None else env
        super().__init__(env, hypers)
        
class _2048MCTS(VectorizedMCTS):
    def __init__(self, num_parallel_envs, device, hypers: MCTSHypers, env: Optional[_2048Env] = None, debug: bool = False) -> None:
        env = _2048Env(num_parallel_envs, device, debug) if env is None else env
        super().__init__(env, hypers)

_2048_EVALUATORS = Union[_2048MCTS, _2048LazyMCTS]
