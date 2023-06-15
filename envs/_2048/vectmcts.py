
import torch
from core.vz_resnet import VZResnet

from core.sp_lazy_mcts import SinglePlayerVectorizedLazyMCTS
from envs._2048.vectenv import _2048Env

class _2048LazyMCTS(SinglePlayerVectorizedLazyMCTS):
    def __init__(self, env: _2048Env, puct_c: int) -> None:
        super().__init__(env, puct_c, 1e5)
        
    