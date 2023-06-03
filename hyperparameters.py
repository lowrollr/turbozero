from dataclasses import dataclass
from typing import Optional


@dataclass()
class AZ_HYPERPARAMETERS:
    learning_rate: float = 5e-5
    minibatch_size: int = 128
    minibatches_per_update: int = 16
    mcts_iters_train: int = 50
    mcts_iters_eval: int = 50
    mcts_c_puct: float = 3.0
    replay_memory_size: int = 1000
    replay_memory_min_size: int = 1000
    policy_factor: int = 1
    episodes_per_epoch: int = 1000
    exploration_cutoff: Optional[int] = None
    epsilon_decay_per_epoch: Optional[float] = None
    num_epochs: int = 100
    eval_games: int = 200
    weight_decay: float = 0.0

@dataclass()
class LazyAZHyperparameters:
    learning_rate: float = 1e-3
    num_iters_train: int = 5
    iter_depth_train: int = 3
    num_iters_eval: int = 5
    iter_depth_test: int = 3
    replay_memory_size: int = 1000
    replay_memory_min_size: int = 1
    policy_factor: int = 1
    minibatch_size: int = 128
    minibatches_per_update: int = 16
    episodes_per_epoch: int = 5
    eval_episodes_per_epoch: int = 5
    epsilon_decay_per_epoch: float = 0.0001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    mcts_c_puct: float = 1.0