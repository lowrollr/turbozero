

import logging

from dataclasses import dataclass
from core.algorithms.evaluator import Evaluator, EvaluatorConfig
from core.env import Env
from core.utils.history import Metric


class Baseline(Evaluator):
    def __init__(self, env: Env, config: EvaluatorConfig, *args, **kwargs):
        super().__init__(env, config, *args, **kwargs)
        self.metrics_key = 'baseline'
        self.proper_name = 'Baseline'

    def add_metrics(self, history):
        if history.epoch_metrics.get(self.metrics_key) is None:
            history.epoch_metrics[self.metrics_key] = Metric(
                name=self.metrics_key, 
                xlabel='Epoch', 
                ylabel='Win Rate', 
                maximize=True, 
                alert_on_best=False, 
                proper_name=f'Win Rate (Current Model vs. {self.proper_name})'
            )
    
    def add_metrics_data(self, data, history, log=True):
        history.add_epoch_data({self.metrics_key: data}, log=log)
        
        
