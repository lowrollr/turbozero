


from typing import Optional
import torch
import numpy as np
from core.history import Metric, TrainingMetrics
from core.hyperparameters import LZHyperparameters
from core.lz_resnet import LZResnet
from core.memory import GameReplayMemory
from core.trainer import Trainer
from envs._2048.collector import _2048Collector
from envs._2048.utils import rotate_training_examples
from envs._2048.vectmcts import _2048LazyMCTS


class _2048Trainer(Trainer):
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
        run_tag: str = '2048',
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


        self.train_collector = _2048Collector(
            _2048LazyMCTS(num_parallel_envs, device, hypers.mcts_c_puct),
            episode_memory_device,
            hypers.num_iters_train,
            hypers.iter_depth_train
        )

        self.test_collector = _2048Collector(
            _2048LazyMCTS(hypers.eval_episodes_per_epoch, device, hypers.mcts_c_puct),
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
            episode_metrics=[
                Metric(name='reward', xlabel='Episode', ylabel='Reward', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results),
                Metric(name='log2_high_square', xlabel='Episode', ylabel='High Tile', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results, proper_name='High Tile (log2)'),
            ],
            eval_metrics=[
                Metric(name='reward', xlabel='Reward', ylabel='Frequency', pl_type='hist',  maximize=True, alert_on_best=False),
                Metric(name='high_square', xlabel='High Tile', ylabel='Frequency Tile', pl_type='bar', maximize=True, alert_on_best=False, proper_name='High Tile'),
            ],
            epoch_metrics=[
                Metric(name='avg_reward', xlabel='Epoch', ylabel='Average Reward', maximize=True, alert_on_best=self.log_results, proper_name='Average Reward'),
                Metric(name='avg_log2_high_square', xlabel='Epoch', ylabel='Average High Tile', maximize=True, alert_on_best=self.log_results, proper_name='Average High Tile (log2)'),
            ]
        )
    
    def add_collection_metrics(self, episodes):
        for episode in episodes:
            moves = len(episode)
            last_state = episodes[-1][0]
            high_square = int(last_state.max().item())
            self.history.add_episode_data({
                'reward': moves,
                'log2_high_square': high_square,
            })

    def add_evaluation_metrics(self, episodes):
        for episode in episodes:
            moves = len(episode)
            last_state = episodes[-1][0]
            high_square = 2 ** int(last_state.max().item())
            self.history.add_evaluation_data({
                'reward': moves,
                'high_square': high_square,
            })

    def add_epoch_metrics(self):
        self.history.add_epoch_data({
            'avg_reward': np.mean(self.history.eval_metrics['reward'][-1].data),
            'avg_log2_high_square': np.log2(np.mean(self.history.eval_metrics['high_square'][-1].data))
        }, log=self.log_results)
    
    def training_step(self):
        inputs, target_policy, target_value = zip(*self.replay_memory.sample(self.hypers.minibatch_size))

        inputs = torch.stack(inputs).to(device=self.device)
        target_policy = torch.stack(target_policy).to(device=self.device)
        target_policy.div_(target_policy.sum(dim=1, keepdim=True))
        target_value = torch.stack(target_value).to(device=self.device).log()

        self.optimizer.zero_grad()
        policy, value = self.model(inputs)

        policy_loss = self.hypers.policy_factor * torch.nn.functional.cross_entropy(policy, target_policy)
        value_loss = torch.nn.functional.mse_loss(value.flatten(), target_value)
        loss = policy_loss + value_loss

        policy_accuracy = torch.eq(torch.argmax(target_policy, dim=1), torch.argmax(policy, dim=1)).float().mean()

        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), policy_accuracy.item(), loss.item()

    


    

def init_2048_trainer_from_checkpoint(
    num_parallel_envs: int,
    checkpoint_path: str,
    device: torch.device,
    episode_memory_device: torch.device = torch.device('cpu'),
    log_results = True,
    interactive = True
) -> _2048Trainer:
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

    return _2048Trainer(
        num_parallel_envs = num_parallel_envs, 
        model = model, 
        optimizer = optimizer, 
        hypers = hypers, 
        device = device,  
        episode_memory_device = episode_memory_device,
        history = history, 
        log_results = log_results, 
        interactive = interactive, 
        run_tag=run_tag
    )