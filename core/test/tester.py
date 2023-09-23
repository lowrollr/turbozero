

from copy import deepcopy
import random
from tqdm import tqdm
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
        debug: bool = False
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.collector = collector
        self.device = self.collector.evaluator.env.device
        self.history = history
        self.log_results = log_results
        self.debug = debug

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
        log_results: bool = True,
        debug: bool = False
    ):
        super().__init__(collector, config, model, history, optimizer, log_results, debug)
        self.config: TwoPlayerTesterConfig
        self.baselines = []
        for baseline_config in config.baselines:
            baseline: Baseline = init_evaluator(baseline_config, collector.evaluator.env, evaluator=collector.evaluator, best_model=model, best_model_optimizer=optimizer)
            self.baselines.append(baseline)
            baseline.add_metrics(self.history)
    
    def collect_test_batch(self):
        for baseline in self.baselines:
            scores = collect_games(self.collector.evaluator, baseline, self.config.episodes_per_epoch, self.device, debug=self.debug)
            wins = (scores == 1).sum().cpu().clone()
            draws = (scores == 0.5).sum().cpu().clone()
            losses = (scores == 0).sum().cpu().clone()
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



def collect_games(evaluator1: Evaluator, evaluator2: Evaluator, num_games: int, device: torch.device, debug: bool) -> torch.Tensor:
    if not debug:
        progress_bar = tqdm(total=num_games, desc='Collecting games...', leave=True, position=0)
    seed = random.randint(0, 2**32 - 1)
    evaluator1.reset(seed)
    evaluator2.reset(seed)
    split = num_games // 2
    reset = torch.zeros(num_games, dtype=torch.bool, device=device, requires_grad=False)
    reset[:split] = True

    completed_episodes = torch.zeros(num_games, dtype=torch.bool, device=device, requires_grad=False)
    scores = torch.zeros(num_games, dtype=torch.float32, device=device, requires_grad=False)

    _, _, _, actions, terminated = evaluator1.step()

    envs_to_reset = terminated | reset

    evaluator1.env.terminated[:split] = True
    evaluator1.env.reset_terminated_states(seed)
    evaluator1.reset_evaluator_states(envs_to_reset)
    evaluator2.step_evaluator(actions, envs_to_reset)

    starting_players = (evaluator1.env.cur_players.clone() - 1) % 2
    use_second_evaluator = True
    while not completed_episodes.all():
        if use_second_evaluator:
            _, _, _, actions, terminated = evaluator2.step()
            evaluator1.step_evaluator(actions, terminated)
        else:
            _, _, _, actions, terminated = evaluator1.step()
            evaluator2.step_evaluator(actions, terminated)
        rewards = evaluator1.env.get_rewards(starting_players)
        scores += rewards * terminated * (~completed_episodes)
        new_completed = (terminated & (~completed_episodes)).long().sum().item()
        completed_episodes |= terminated
        evaluator1.env.reset_terminated_states(seed)
        use_second_evaluator = not use_second_evaluator
        if not debug:
            progress_bar.update(new_completed)

    return scores
