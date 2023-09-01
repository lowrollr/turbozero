
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from core.env import Env
from core.utils.utils import rand_argmax_2d

@dataclass
class EvaluatorConfig:
    name: str

class Evaluator:
    def __init__(self, env: Env, config: EvaluatorConfig, *args, **kwargs):
        self.device = env.device
        self.env = env
        self.env.reset()
        self.config = config
        self.epsilon = 1e-8
        self.args = args
        self.kwargs = kwargs

    def reset(self, seed=None) -> int:
        return self.env.reset(seed=seed)

    def evaluate(self, *args) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # returns probability distribution over actions, and optionally the value of the current state
        raise NotImplementedError()
    
    def step_evaluator(self, actions, terminated) -> None:
        pass

    def step_env(self, actions) -> torch.Tensor:
        terminated = self.env.step(actions)
        self.step_evaluator(actions, terminated)
        return terminated

    def choose_actions(self, probs: torch.Tensor) -> torch.Tensor:
        legal_actions = self.env.get_legal_actions()
        return rand_argmax_2d((probs + self.epsilon) * legal_actions).flatten()

    def step(self, *args) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        initial_states = self.env.get_nn_input().clone()
        probs, values = self.evaluate(*args)
        actions = self.choose_actions(probs)
        terminated = self.step_env(actions)
        return initial_states, probs, values, actions, terminated
    
    def reset_evaluator_states(self, evals_to_reset: torch.Tensor) -> None:
        pass

class TrainableEvaluator(Evaluator):
    def __init__(self, env: Env, config: EvaluatorConfig, model: torch.nn.Module):
        super().__init__(env, config)
        self.env = env
        self.config = config
        self._model = model

    @property
    def model(self) -> torch.nn.Module:
        return self._model