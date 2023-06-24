
import torch

from core import GLOB_FLOAT_TYPE
from .vectenv import VectEnv

class VectorizedLazyMCTS:
    def __init__(self, env: VectEnv, puct_coeff: float, very_positive_value: float = 1e8) -> None:
        self.env = env

        self.action_scores = torch.zeros(
            (self.env.num_parallel_envs, *self.env.policy_shape), 
            dtype=GLOB_FLOAT_TYPE,
            device=self.env.device,
            requires_grad=False
        )

        self.visit_counts = torch.zeros_like(
            self.action_scores, 
            dtype=GLOB_FLOAT_TYPE, 
            device=self.env.device, 
            requires_grad=False
        )


        self.puct_coeff = puct_coeff

        # TODO: this is not an elegant solution at all but we need a large, positive, non-infinite value
        # this requires implementations to think carefully about choosing this value, which should be unnecessary
        self.very_positive_value = very_positive_value


    def reset(self, seed=None) -> None:
        self.env.reset(seed=seed)
        self.reset_puct()
    
    def reset_puct(self) -> None:
        self.action_scores.zero_()
        self.visit_counts.zero_()

    def choose_action_with_puct(self, probs: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def iterate(self, model: torch.nn.Module, depth: int) -> torch.Tensor:
        raise NotImplementedError()
    
    def explore(self, model: torch.nn.Module, iters: int, search_depth: int) -> torch.Tensor:
        self.reset_puct()

        return self.explore_for_iters(model, iters, search_depth)
    
    def explore_for_iters(self, model: torch.nn.Module, iters: int, search_depth: int) -> torch.Tensor:
        raise NotImplementedError()