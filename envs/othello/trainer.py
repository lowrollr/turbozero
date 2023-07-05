




from copy import deepcopy
from pathlib import Path
from typing import Optional
import torch
import logging
from core.utils.history import Metric, TrainingMetrics
from core.training.training_hypers import TurboZeroHypers
from core.resnet import TurboZeroResnet
from core.utils.memory import GameReplayMemory
from core.training.trainer import Trainer
from envs.othello.collector import OthelloCollector
from envs.othello.evaluator import OTHELLO_EVALUATORS


class OthelloTrainer(Trainer):
    def __init__(self,
        evaluator_train: OTHELLO_EVALUATORS,
        evaluator_test: OTHELLO_EVALUATORS,
        num_parallel_envs: int,
        device: torch.device,
        episode_memory_device: torch.device,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        hypers: TurboZeroHypers,
        history: Optional[TrainingMetrics] = None,
        log_results: bool = True,
        interactive: bool = True,
        run_tag: str = 'othello'
    ):
        train_collector = OthelloCollector(
            evaluator_train,
            episode_memory_device,
        )
        test_collector = OthelloCollector(
            evaluator_test,
            episode_memory_device,
        )
        super().__init__(
            train_collector = train_collector,
            test_collector = test_collector,
            num_parallel_envs = num_parallel_envs,
            model = model,
            optimizer = optimizer,
            hypers = hypers,
            device = device,
            history = history,
            log_results = log_results,
            interactive = interactive,
            run_tag = run_tag
        )

        self.best_model = deepcopy(model)
        self.random_baseline = deepcopy(model)
        self.best_model_optimizer_state_dict = deepcopy(optimizer.state_dict())

    def save_checkpoint(self, custom_name: Optional[str] = None) -> None:
        directory = f'./checkpoints/{self.run_tag}/'
        Path(directory).mkdir(parents=True, exist_ok=True)
        filename = custom_name if custom_name is not None else str(self.history.cur_epoch)
        filepath = directory + f'{filename}.pt'
        torch.save({
            'model_arch_params': self.model.arch_params,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_model_state_dict': self.best_model.state_dict(),
            'best_model_optimizer_state_dict': self.best_model_optimizer_state_dict,
            'hypers': self.hypers,
            'history': self.history,
            'run_tag': self.run_tag
        }, filepath)

    def init_history(self):
        return TrainingMetrics(
            train_metrics=[
                Metric(name='loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results),
                Metric(name='value_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Value Loss'),
                Metric(name='policy_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Policy Loss'),
                Metric(name='policy_accuracy', xlabel='Step', ylabel='Accuracy (%)', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results, proper_name='Policy Accuracy'),
            ],
            episode_metrics=[],
            eval_metrics=[],
            epoch_metrics=[
                Metric(name='win_margin_vs_best', xlabel='Epoch', ylabel='Margin (+/- games)', maximize=True, alert_on_best=False, proper_name='Win Margin (Current Model vs. Best Model)'),
                Metric(name='win_margin_vs_random', xlabel='Epoch', ylabel='Margin (+/- games)', maximize=True, alert_on_best=False, proper_name='Win Margin (Current Model vs. Random Model)'),
            ]
        )
    
    def add_collection_metrics(self, episodes):
        self.history.add_episode_data({}, log=self.log_results)
    def add_evaluation_metrics(self, episodes):
        self.history.add_evaluation_data({}, log=self.log_results)
    def add_epoch_metrics(self):
        pass

    def evaluate_against(self, num_episodes, other_model):
        self.test_collector.reset()
        split = num_episodes // 2
        completed_episodes = torch.zeros(num_episodes, dtype=torch.bool, device=self.device, requires_grad=False)
        scores = torch.zeros(num_episodes, dtype=torch.float32, device=self.device, requires_grad=False)
        self.test_collector.collect_step(self.model, epsilon=0.0)
        # hacky way to split the episodes into two sets (this environment cannot terminate on the first step)
        self.test_collector.evaluator.env.terminated[:split] = True
        self.test_collector.evaluator.env.reset_terminated_states()
        use_other_model = True
        while not completed_episodes.all():
            model = other_model if use_other_model else self.model
            # we don't need to collect the episodes into episode memory/replay buffer, so we can call collect_step directly
            terminated = self.test_collector.collect_step(model, epsilon=0.0)

            if use_other_model:
                # rewards are from the perspective of the next player
                scores += self.test_collector.evaluator.env.get_rewards() * terminated * ~completed_episodes
            else:
                scores += (1 - self.test_collector.evaluator.env.get_rewards()) * terminated * ~completed_episodes

            completed_episodes |= terminated
            use_other_model = not use_other_model

        wins = (scores == 1).sum().cpu().clone()
        draws = (scores == 0.5).sum().cpu().clone()
        losses = (scores == 0).sum().cpu().clone()

        return wins, draws, losses


    def evaluate_n_episodes(self, num_episodes):
        
        wins, draws, losses = self.evaluate_against(num_episodes, self.best_model)
        win_margin_vs_best = wins - losses
        new_best = win_margin_vs_best >= self.hypers.test_improvement_threshold
        logging.info(f'Epoch {self.history.cur_epoch} Current vs. Best:')
        if new_best:
            self.best_model.load_state_dict(self.model.state_dict())
            self.best_model_optimizer_state_dict = deepcopy(self.optimizer.state_dict())
            logging.info('************ NEW BEST MODEL ************')
        else:
            self.model.load_state_dict(self.best_model.state_dict())
            self.optimizer.load_state_dict(self.best_model_optimizer_state_dict)
        logging.info(f'W/L/D: {wins}/{losses}/{draws}')
        
        
        
        wins, draws, losses = self.evaluate_against(num_episodes, self.random_baseline)
        win_margin_vs_random = wins - losses
        logging.info(f'Epoch {self.history.cur_epoch} Current vs. Random:')
        logging.info(f'W/L/D: {wins}/{losses}/{draws}')
        self.history.add_epoch_data({
            'win_margin_vs_best': win_margin_vs_best,
            'win_margin_vs_random': win_margin_vs_random,
        }, log=self.log_results)
        self.add_evaluation_metrics([])



def load_checkpoint(
    num_parallel_envs: int,
    checkpoint_path: str,
    device: torch.device,
    episode_memory_device: torch.device = torch.device('cpu'),
    log_results = True,
    interactive = True,
    debug = False,
) -> OthelloTrainer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hypers: TurboZeroHypers = checkpoint['hypers']
    model = TurboZeroResnet(checkpoint['model_arch_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hypers.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    history = checkpoint['history']
    history.reset_all_figs()
    run_tag = checkpoint['run_tag']
    train_hypers = checkpoint['train_evaluator']['hypers']
    train_evaluator: OTHELLO_EVALUATORS = checkpoint['train_evaluator']['type'](num_parallel_envs, device, 8, train_hypers, debug=debug)
    test_hypers = checkpoint['test_evaluator']['hypers']
    test_evaluator: OTHELLO_EVALUATORS = checkpoint['test_evaluator']['type'](num_parallel_envs, device, 8, test_hypers, debug=debug)
    return OthelloTrainer(train_evaluator, test_evaluator, num_parallel_envs, device, episode_memory_device, model, optimizer, hypers, history, log_results, interactive, run_tag)



