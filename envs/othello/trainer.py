




from typing import Optional
import torch
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
            epoch_metrics=[]
        )
    
    def add_collection_metrics(self, episodes):
        pass
    def add_evaluation_metrics(self, episodes):
        pass
    def add_epoch_metrics(self, episodes):
        pass

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

    
