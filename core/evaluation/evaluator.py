
import torch
from core.vectenv import VectEnv


class Evaluator:
    def __init__(self, env: VectEnv, device: torch.device, hypers):
        self.device = device
        self.env = env
        self.env.reset()
        self.hypers = hypers

    def reset(self, seed=None) -> None:
        self.env.reset(seed=seed)

    def evaluate(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def step_env(self, actions) -> torch.Tensor:
        return self.env.step(actions)
