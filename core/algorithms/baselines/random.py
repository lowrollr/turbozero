




from typing import Optional, Tuple
import torch
from core.algorithms.baselines.baseline import Baseline, BaselineConfig
from core.algorithms.evaluator import Evaluator
from core.env import Env
from core.utils.history import Metric


class RandomBaseline(Baseline):
    def __init__(self, 
        env: Env, 
        config: BaselineConfig,
        metrics_key: str = 'random',
        proper_name: str = 'Random',
        **kwargs
    ) -> None:
        super().__init__(env, config)
        self.metrics_key = metrics_key
        self.proper_name = proper_name
        self.sample = torch.zeros(self.env.parallel_envs, self.env.policy_shape[0], device=self.device, requires_grad=False, dtype=torch.float32)

    def evaluate(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: 
        self.sample.uniform_(0, 1)
        return self.sample, None