
import torch
from core.lz_resnet import LZResnet

from core.lazy_mcts import VectorizedLazyMCTS
from envs._2048.vectenv import _2048Env

class _2048LazyMCTS(VectorizedLazyMCTS):
    def __init__(self, num_parallel_envs, device, puct_c: float) -> None:
        env = _2048Env(num_parallel_envs, device)
        super().__init__(env, puct_c, 1e5)
        
    