

import logging

from dataclasses import dataclass
from core.algorithms.evaluator import Evaluator, EvaluatorConfig
from core.utils.history import Metric

@dataclass
class BaselineConfig(EvaluatorConfig):
    name: str


class Baseline(Evaluator):
    def __init__(self, env, device, config):
        super().__init__(env, device, config)
        self.metrics_key = 'baseline'
        self.proper_name = 'Baseline'

    def add_metrics(self, history):
        history.epoch_metrics[self.metrics_key] = Metric(
            name=self.metrics_key, 
            xlabel='Epoch', 
            ylabel='Margin (+/- games)', 
            maximize=True, 
            alert_on_best=False, 
            proper_name=f'Win Margin (Current Model vs. {self.proper_name})'
        )
    
    def add_metrics_data(self, data, history, log=True):
        history.add_epoch_data({self.metrics_key: data}, log=log)
        
        
