




from copy import deepcopy
from pathlib import Path
from typing import Optional
import torch
import logging
from core.history import Metric, TrainingMetrics
from core.hyperparameters import LZHyperparameters
from core.lz_resnet import LZResnet
from core.memory import GameReplayMemory
from core.trainer import Trainer
from envs.othello.collector import OthelloCollector
from envs.othello.vectmcts import OthelloLazyMCTS


class OthelloTrainer(Trainer):
    def __init__(self,
        num_parallel_envs: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        hypers: LZHyperparameters,
        device: torch.device,
        episode_memory_device: torch.device = torch.device('cpu'),
        history: Optional[TrainingMetrics] = None,
        log_results: bool = True,
        interactive: bool = True,
        run_tag: str = 'othello',
        board_size: int = 8,
        debug = False
    ):
        super().__init__(
            num_parallel_envs,
            model,
            optimizer,
            hypers,
            device,
            episode_memory_device,
            history,
            log_results,
            interactive,
            run_tag,
        )

        self.train_collector = OthelloCollector(
            OthelloLazyMCTS(num_parallel_envs, device, board_size, hypers.mcts_c_puct, debug=debug),
            episode_memory_device,
            hypers.num_iters_train,
            hypers.iter_depth_train
        )

        self.test_collector = OthelloCollector(
            OthelloLazyMCTS(hypers.eval_episodes_per_epoch, device, board_size, hypers.mcts_c_puct, debug=debug),
            episode_memory_device,
            hypers.num_iters_eval,
            hypers.iter_depth_test
        )

        self.replay_memory = GameReplayMemory(
            hypers.replay_memory_size
        )

        self.best_model = deepcopy(model)
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
                Metric(name='win_margin', xlabel='Epoch', ylabel='Margin (+/- games)', maximize=True, alert_on_best=False, proper_name='Win Margin (Current Model vs. Best Model)'),
            ]
        )
    
    def add_collection_metrics(self, episodes):
        self.history.add_episode_data({}, log=self.log_results)
    def add_evaluation_metrics(self, episodes):
        self.history.add_evaluation_data({}, log=self.log_results)
    def add_epoch_metrics(self):
        pass

    def evaluate_n_episodes(self, num_episodes):
        self.test_collector.reset()
        # pits the current model against the previous best model
        split = num_episodes // 2
        completed_episodes = torch.zeros(num_episodes, dtype=torch.bool, device=self.device, requires_grad=False)
        scores = torch.zeros(num_episodes, dtype=torch.float32, device=self.device, requires_grad=False)
        self.test_collector.collect_step(self.model, epsilon=0.0)
        # hacky way to split the episodes into two sets (this environment cannot terminate on the first step)
        self.test_collector.evaluator.env.terminated[:split] = True
        self.test_collector.evaluator.env.reset_terminated_states()
        use_best_model = True
        while not completed_episodes.all():
            model = self.best_model if use_best_model else self.model
            # we don't need to collect the episodes into episode memory/replay buffer, so we can call collect_step directly
            terminated = self.test_collector.collect_step(model, epsilon=0.0)

            if use_best_model:
                # rewards are from the perspective of the next player
                scores += self.test_collector.evaluator.env.get_rewards() * terminated * ~completed_episodes
            else:
                scores += (1 - self.test_collector.evaluator.env.get_rewards()) * terminated * ~completed_episodes

            completed_episodes |= terminated
            use_best_model = not use_best_model

        wins = (scores == 1).sum().cpu().clone()
        draws = (scores == 0.5).sum().cpu().clone()
        losses = (scores == 0).sum().cpu().clone()
        win_margin = wins - losses

        new_best = win_margin >= self.hypers.improvement_threshold

        if new_best:
            self.best_model.load_state_dict(self.model.state_dict())
            self.best_model_optimizer_state_dict = deepcopy(self.optimizer.state_dict())
            logging.info('************ NEW BEST MODEL ************')
            logging.info(f'Epoch {self.history.cur_epoch} - W/L/D: {wins}/{losses}/{draws}')
        else:
            self.model.load_state_dict(self.best_model.state_dict())
            self.optimizer.load_state_dict(self.best_model_optimizer_state_dict)
            logging.info(f'Epoch {self.history.cur_epoch} - W/L/D: {wins}/{losses}/{draws}')
        self.add_evaluation_metrics([])
        self.history.add_epoch_data({
            'win_margin': win_margin,
        }, log=self.log_results)
        

def init_2048_trainer_from_checkpoint(
    num_parallel_envs: int,
    checkpoint_path: str,
    device: torch.device,
    episode_memory_device: torch.device = torch.device('cpu'),
    log_results = True,
    interactive = True,
    board_size: int = 8
) -> OthelloTrainer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hypers: LZHyperparameters = checkpoint['hypers']
    model = LZResnet(checkpoint['model_arch_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hypers.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    history = checkpoint['history']
    history.reset_all_figs()
    run_tag = checkpoint['run_tag']

    return OthelloTrainer(
        num_parallel_envs = num_parallel_envs, 
        model = model, 
        optimizer = optimizer, 
        hypers = hypers, 
        device = device,  
        episode_memory_device = episode_memory_device,
        history = history, 
        log_results = log_results, 
        interactive = interactive, 
        run_tag = run_tag,
        board_size = board_size
    )

    
