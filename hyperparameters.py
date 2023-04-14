from dataclasses import dataclass


@dataclass()
class AZ_HYPERPARAMETERS:
    learning_rate: float = 1e-4
    minibatch_size: int = 128
    minibatches_per_update: int = 16
    mcts_iters_train: int = 50
    mcts_iters_eval: int = 50
    mcts_c_puct: float = 1.0
    replay_memory_size: int = 1000
    replay_memory_min_size: int = 1000
    policy_factor: int = 50
    episodes_per_epoch: int = 500
    num_epochs: int = 100
    eval_games: int = 100
    tau_s: float = 1.5
    tau_m: float = 0.49
    tau_b: float = 0.50
    tau_d: float = 10.0
    weight_decay: float = 0.0
