# TurboZero
The TurboZero project contains vectorized, hardware-accelerated implementations of AlphaZero-esque algorithms, alongside vectorized implementations of single-player and multi-player environments. Basic training infrastructure is also included, which means models can be trained for supported environments straight out of the box. This project is similar to DeepMind's [mctx](https://github.com/deepmind/mctx), but as of now is more focused on model-based algorithms like AlphaZero rather than model-free implementations such as MuZero, and is written with PyTorch instead of JAX. Due to this focus, TurboZero includes additional features relavant to model-based algorithms, such as persisting MCTS subtrees. I hope to eventually expand this project and implemented accelerated adaptations of other RL algorithms, like PPO, MuZero, etc.

## Motivation 
Training reinforcement learning algorithms is notoriously compute-intensive. Oftentimes models must train for millions of episodes to reach desired performance, with each episode containing many steps and each step requiring numerous model inference calls and dynamic game-tree exploration. All of these factors contribute to RL training tasks sometimes being prohibitvely expensive, even when taking advantage of process (CPU) parallelism. However, if environments and infrastructure can be implemented as a set of multi-dimensional matrix operations, this computation can be offloaded to GPUs, reaping all the benefits of GPU parallelism by training on and evaluating stacked environments in parallel. TurboZero includes implementations of simulation environments and Rl algorithms that do just that.

While other common open-source implementations of AlphaZero complete training runs in days/weeks, TurboZero can complete similar tasks in minutes/hours when paired with the appropriate hardware.



## Supported Environments

| Environment | Type | Observation Size | Policy Size | Description | 
| --- | --- | --- | --- | --- |
| Othello | Multi-Player |2x8x8 | 65 | 2-player tile-swapping game played on an 8x8 board. also called Reversi |
| 2048 | Single-Player |4x4 | 4 | Single-player numeric puzzle game |
| ConnectX | WIP |
| Chess | WIP |
| Go | WIP | 

## Supported Algorithms

| Name | Description | Hyperparameters | Paper |
| --- | --- | --- | --- |
| AlphaZero | DeepMind's algorithm that first famously defeated Lee Sodol in Go and has since been shown to generalize well to other games such as Chess and Shogi as well as more sophisticated tasks like code generation and video compression. | [dataclass](https://github.com/lowrollr/lazyzero/blob/main/core/evaluation/mcts_hypers.py) | [Silver, 2017](https://arxiv.org/abs/1712.01815)
| LazyZero | A lazy implementation of AlphaZero that only utilizes PUCT to dictate exploration at the root node. Exploration steps instead use fixed depth rollouts sampling from the trained model policy. I wrote this as an easier-to-implement alternative to AlphaZero, and showed it can effectively train models to beat 2048. | [dataclass](https://github.com/lowrollr/lazyzero/blob/main/core/evaluation/lazy_mcts_hypers.py) | | 

## Quickstart

## Future Work

## Contributing


## About the Authors


## Cite This Work