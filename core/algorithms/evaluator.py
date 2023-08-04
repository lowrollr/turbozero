
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from core.env import Env

@dataclass
class EvaluatorConfig:
    # nothing to see here
    pass

class Evaluator:
    def __init__(self, env: Env, device: torch.device, config: EvaluatorConfig):
        self.device = device
        self.env = env
        self.env.reset()
        self.config = config

    def reset(self, seed=None) -> None:
        self.env.reset(seed=seed)

    def evaluate(self, *args) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # returns probability distribution over actions, and optionally the value of the current state
        raise NotImplementedError()

    def step_env(self, actions) -> torch.Tensor:
        return self.env.step(actions)

    def choose_actions(self, probs: torch.Tensor) -> torch.Tensor:
        legal_actions = self.env.get_legal_actions()
        return torch.argmax(probs * legal_actions, dim=1)

    def step(self, *args) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        initial_states = self.env.states.clone()
        probs, values = self.evaluate(*args)
        actions = self.choose_actions(probs)
        terminated = self.step_env(actions)
        return initial_states, probs, values, actions, terminated

