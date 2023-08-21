

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from core.algorithms.baselines.baseline import Baseline
from core.algorithms.evaluator import Evaluator, EvaluatorConfig
from core.env import Env
from core.utils.history import Metric

class BestModelBaseline(Baseline):
    def __init__(self, 
        env: Env,
        config: EvaluatorConfig, 
        evaluator: Evaluator, 
        best_model: torch.nn.Module,
        best_model_optimizer: torch.optim.Optimizer,
        metrics_key: str = 'win_rate_vs_best',
        proper_name: str = 'Best Model',
        *args, **kwargs
    ):
        super().__init__(env, config, *args, **kwargs)
        self.best_model = deepcopy(best_model)
        self.best_model_optimizer = deepcopy(best_model_optimizer.state_dict()) if best_model_optimizer is not None else None
        self.evaluator = evaluator.__class__(env, evaluator.config, self.best_model)
        self.metrics_key = metrics_key
        self.proper_name = proper_name

    def step_evaluator(self, actions, terminated) -> None:
        self.evaluator.step_evaluator(actions, terminated)

    def step(self, *args) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        return self.evaluator.step(self.best_model)