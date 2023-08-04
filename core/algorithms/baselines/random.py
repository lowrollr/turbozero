




import torch
from core.algorithms.baselines.baseline import Baseline, BaselineConfig
from core.algorithms.evaluator import Evaluator
from core.env import Env
from core.utils.history import Metric


class RandomBaseline(Baseline):
    def __init__(self, 
        env: Env, 
        device: torch.device, 
        config: BaselineConfig,
        metrics_key: str = 'random',
        proper_name: str = 'Random',
        **kwargs
    ) -> None:
        super().__init__(env, device, config)
        self.metrics_key = metrics_key
        self.proper_name = proper_name

    def evaluate(self):
        legal_actions = self.env.get_legal_actions().float()
        return torch.multinomial(legal_actions, 1, replacement=True).flatten()