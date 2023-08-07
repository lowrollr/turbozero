

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from core.algorithms.baselines.baseline import Baseline, BaselineConfig
from core.algorithms.evaluator import Evaluator, EvaluatorConfig
from core.env import Env
from core.utils.history import Metric

@dataclass
class BestModelBaselineConfig(BaselineConfig):
    pass

class BestModelBaseline(Baseline):
    def __init__(self, 
        env: Env,
        config: BestModelBaselineConfig, 
        evaluator: Evaluator, 
        best_model: torch.nn.Module,
        best_model_optimizer: torch.optim.Optimizer,
        metrics_key: str = 'win_margin_vs_best',
        proper_name: str = 'Best Model',
        **kwargs
    ):
        super().__init__(env, config)
        self.best_model = deepcopy(best_model)
        self.best_model_optimizer = deepcopy(best_model_optimizer.state_dict()) if best_model_optimizer is not None else None
        self.evaluator = evaluator.__class__(env, evaluator.config)
        self.metrics_key = metrics_key
        self.proper_name = proper_name

    def evaluate(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.evaluator.evaluate(self.best_model)

    
    
