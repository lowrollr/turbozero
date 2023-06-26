


from core.lazy_mcts import VectorizedLazyMCTS
from envs.othello.vectenv import OthelloVectEnv


class OthelloLazyMCTS(VectorizedLazyMCTS):
    def __init__(self, env: OthelloVectEnv, puct_c: int) -> None:
        super().__init__(env, puct_c, 1e5)