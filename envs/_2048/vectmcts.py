
import torch
from core.lz_resnet import LZResnet

from core.lazy_mcts import VectorizedLazyMCTS
from envs._2048.vectenv import _2048Env

class _2048LazyMCTS(VectorizedLazyMCTS):
    def __init__(self, env: _2048Env, puct_c: float) -> None:
        super().__init__(env, puct_c, 1e5)
        
    