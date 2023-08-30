

from copy import deepcopy
import torch
from typing import List, Optional
from core.algorithms.baselines.baseline import Baseline
from core.algorithms.baselines.best import BestModelBaseline
from core.algorithms.evaluator import Evaluator, EvaluatorConfig
from core.algorithms.load import init_evaluator
from core.train.collector import Collector
from core.utils.history import TrainingMetrics
import logging 

from dataclasses import dataclass

@dataclass
class TesterConfig:
    algo_config: EvaluatorConfig
    episodes_per_epoch: int

class Tester:
    def __init__(self, 
        collector: Collector, 
        config: TesterConfig,
        model: torch.nn.Module,
        history: TrainingMetrics,
        optimizer: Optional[torch.optim.Optimizer] = None,
        log_results: bool = True,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.collector = collector
        self.device = self.collector.evaluator.env.device
        self.history = history
        self.log_results = log_results

    def add_evaluation_metrics(self, episodes):
        raise NotImplementedError()

    def collect_test_batch(self):
        self.collector.reset()
        # we make a big assumption here that all evaluation episodes can be run in parallel
        # if the user wants to evaluate an obscene number of episodes this will become problematic
        # TODO: batch evaluation episodes where necessary
        completed_episodes = torch.zeros(self.config.episodes_per_epoch, dtype=torch.bool, device=self.collector.evaluator.env.device)
        while not completed_episodes.all():
            episodes, termianted = self.collector.collect(inactive_mask=completed_episodes)
            self.add_evaluation_metrics(episodes)
            completed_episodes |= termianted
    
    def generate_plots(self):
        self.history.generate_plots(categories=['eval'])
        
    
@dataclass
class TwoPlayerTesterConfig(TesterConfig):
    baselines: List[dict]
    improvement_threshold_pct: float = 0.0

    
class TwoPlayerTester(Tester):
    def __init__(self, 
        collector: Collector, 
        config: TwoPlayerTesterConfig,
        model: torch.nn.Module, 
        history: TrainingMetrics,
        optimizer: Optional[torch.optim.Optimizer] = None,
        log_results: bool = True
    ):
        super().__init__(collector, config, model, history, optimizer, log_results)
        self.config: TwoPlayerTesterConfig
        self.baselines = []
        for baseline_config in config.baselines:
            baseline: Baseline = init_evaluator(baseline_config, collector.evaluator.env, evaluator=collector.evaluator, best_model=model, best_model_optimizer=optimizer)
            self.baselines.append(baseline)
            baseline.add_metrics(self.history)

    def evaluate_against_baseline(self, baseline: Evaluator):
        self.collector.reset()
        split = self.config.episodes_per_epoch // 2
        reset = torch.zeros(self.config.episodes_per_epoch, dtype=torch.bool, device=self.collector.evaluator.env.device, requires_grad=False)
        reset[:split] = True
        completed_episodes = torch.zeros(self.config.episodes_per_epoch, dtype=torch.bool, device=self.collector.evaluator.env.device, requires_grad=False)
        scores = torch.zeros(self.config.episodes_per_epoch, dtype=torch.float, device=self.collector.evaluator.env.device, requires_grad=False)
        actions, terminated = self.collector.collect_step()
        envs_to_reset = terminated | reset
        
        self.collector.evaluator.env.terminated[:split] = True
        self.collector.evaluator.env.reset_terminated_states()
        self.collector.evaluator.reset_terminated_envs(envs_to_reset)
        starting_players = (self.collector.evaluator.env.cur_players.clone() - 1) % self.collector.evaluator.env.num_players
        use_other_evaluator = True
        while not completed_episodes.all():
            if use_other_evaluator:
                _, _, _, actions, terminated = baseline.step()
                self.collector.evaluator.step_evaluator(actions, terminated)
            else:
                actions, terminated = self.collector.collect_step()
                baseline.step_evaluator(actions, terminated)
            rewards = self.collector.evaluator.env.get_rewards(starting_players)
            scores += rewards * terminated * (~completed_episodes)
            completed_episodes |= terminated
            use_other_evaluator = not use_other_evaluator

        wins = (scores == 1).sum().cpu().clone()
        draws = (scores == 0.5).sum().cpu().clone()
        losses = (scores == 0).sum().cpu().clone()

        return wins, draws, losses
    
    def collect_test_batch(self):
        for baseline in self.baselines:
            wins, draws, losses = self.evaluate_against_baseline(baseline)
            win_pct = wins / self.config.episodes_per_epoch
            if isinstance(baseline, BestModelBaseline):
                if win_pct > self.config.improvement_threshold_pct:
                    baseline.best_model = deepcopy(self.model)
                    baseline.best_model_optimizer = deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None
                    logging.info('************ NEW BEST MODEL ************')
            if self.history:
                baseline.add_metrics_data(win_pct, self.history, log=self.log_results)
                logging.info(f'Epoch {self.history.cur_epoch} Current vs. {baseline.proper_name}:')
                logging.info(f'W/L/D: {wins}/{losses}/{draws}')



        
