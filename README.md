

![alphazero turns the tide!](./misc/othello_game.gif) ![2048](./misc/2048.gif)
# 🏁 TurboZero 
The TurboZero project contains vectorized, hardware-accelerated implementations of AlphaZero-esque algorithms, alongside vectorized implementations of single-player and multi-player environments. Basic training infrastructure is also included, which means models can be trained for supported environments straight out of the box. This project is similar to DeepMind's [mctx](https://github.com/deepmind/mctx), but as of now is more focused on model-based algorithms like AlphaZero rather than model-free implementations such as MuZero, and is written with PyTorch instead of JAX. Due to this focus, TurboZero includes additional features relavant to model-based algorithms, such as persisting MCTS subtrees. I hope to eventually expand this project and implemented hardware-accelerated adaptations of other RL algorithms, like *MuZero* and *Stochastic AlphaZero*. 

This project has been a labor of love but is still a little rough around the edges. I've done my best to fully explain all configuration options in this file as well as in the [wiki](https://github.com/lowrollr/turbozero/wiki). The [wiki](https://github.com/lowrollr/turbozero/wiki) also provides notes on implementation and vectorization for each of the environments as well as Monte Carlo Tree Search. While as of writing this I believe the project is in a usable, useful state, I still intend to do a great deal of work expanding functionaltiy, fixing issues, and improving performance. I cannot garauntee that data models or workflows will not drastically change as the project matures.

## Motivation 
Training reinforcement learning algorithms is notoriously compute-intensive. Oftentimes models must train for millions of episodes to reach desired performance, with each episode containing many steps and each step requiring numerous model inference calls and dynamic game-tree exploration. All of these factors contribute to RL training tasks sometimes being prohibitvely expensive, even when taking advantage of process (CPU) parallelism. However, if environments and algorithms can be implemented as a set of multi-dimensional matrix operations, this computation can be offloaded to GPUs, reaping all the benefits of GPU parallelism by training on and evaluating stacked environments in parallel. TurboZero includes implementations of simulation environments and RL algorithms that do just that.

While other common open-source implementations of AlphaZero complete training runs in days/weeks, TurboZero can complete similar tasks in minutes/hours when paired with the appropriate hardware.

Vectorized environments are available across a variety of projects at this point. TurboZero's main contribution, therefore, is its vectorized implementaiton of MCTS that supports subtree persistence, which is integrated into a feature-rich RL training pipeline with minimal effort. One direction I'd like to go in the future is integrating with 3rd-party vectorized environments, as I believe this would dramatically increase TurboZero's usefulness.

## Features
### Environments
TurboZero provides vectorized implementations of the following environments:
| Environment | Type | Observation Size | Policy Size | Description | 
| --- | --- | --- | --- | --- |
| [Othello](https://github.com/lowrollr/turbozero/wiki/Othello-Env) | Multi-Player |2x8x8 | 65 | 2-player tile-swapping game played on an 8x8 board. also called Reversi |
| [2048](https://github.com/lowrollr/turbozero/wiki/2048-Env) | Single-Player |4x4 | 4 | Single-player numeric puzzle game |

Each environment supports the full suite of training and evaluation tools, and are implemented with GPU-acceleration in mind. Links to the environment readmes are found above, which provide information on configuration options, implementation details, and results acheived.

### Training
TurboZero supports training policy/value models via the following vectorized algorithms:
| Name | Description | Hyperparameters | Paper |
| --- | --- | --- | --- |
| [AlphaZero](https://github.com/lowrollr/turbozero/wiki/Vectorized-AlphaZero) | DeepMind's algorithm that first famously defeated Lee Sodol in Go and has since been shown to generalize well to other games such as Chess and Shogi as well as more sophisticated tasks like code generation and video compression. | [hyperparameters](https://github.com/lowrollr/turbozero/wiki/Vectorized-AlphaZero#parameters) | [Silver et al., 2017](https://arxiv.org/abs/1712.01815)
| [LazyZero](https://github.com/lowrollr/turbozero/wiki/LazyZero) | A lazy implementation of AlphaZero that only utilizes PUCT to dictate exploration at the root node. Exploration steps instead use fixed depth rollouts sampling from the trained model policy. I wrote this as a simpler, albeit worse alternative to AlphaZero, and showed it can effectively train models to play *2048* and win. | [hyperparameters](https://github.com/lowrollr/turbozero/wiki/LazyZero#configuration-parameters) | | 

Training can be done in a Jupyter notebook, or via the command-line. In addition to environment parameters and training hyperparameters, the user may specify the number of environments to train in parallel, so that the user is able to optimize for their own hardware. See [Quickstart](https://github.com/lowrollr/turbozero#quickstart) for a quick guide on how to get started, or [Training](https://github.com/lowrollr/turbozero/wiki/Training) for full information on configurating your training run. I also provide example configurations that I have used to train effective models for each environment.  

### Evaluation
In addition to the algorithms supporting training a policy, TurboZero also provides vectorized implementations of the following algorithms that serve as baselines to evaluate against:
| Name | Description | Parameters | 
| --- | --- | --- | 
| [Greedy MCTS](https://github.com/lowrollr/turbozero/wiki/Evaluation-&-Testing#greedy-mcts) | MCTS using a heurisitc function to evaluate leaf nodes | [parameters](https://github.com/lowrollr/turbozero/wiki/Evaluation-&-Testing#parameters-2) |
| [Greedy](https://github.com/lowrollr/turbozero/wiki/Evaluation-&-Testing#greedy) | Evaluates potential actions using a heuristic function, no tree search | [parameters](https://github.com/lowrollr/turbozero/wiki/Evaluation-&-Testing#parameters-1)
| [Random](https://github.com/lowrollr/turbozero/wiki/Evaluation-&-Testing#random) | Makes a random legal move | [parameters](https://github.com/lowrollr/turbozero/wiki/Evaluation-&-Testing#parameters)

Evaluating against these algorithms can be baked into the evaluation step of a training run, or be run independently. See [Evaluation & Testing](https://github.com/lowrollr/turbozero/wiki/Evaluation-&-Testing) for the full configuration specification.

### Tournaments / Calculating Elo

Available for multi-player environments, tournaments provide a great way to gauge the relative strength of an algorithm in relation to various opponents. This allows the user to evaluate the effectiveness of adjusting parameters of an algorithm, or analyze how effective increasing the size of a neural network is in terms of performance. In addition, tournaments allow algorithms to be compared against a large cohort of baseline algorithms. Where applicable, I provide tournament data for each environment that will allow you to test your algorithms and models against a pre-populated field. 

For more about tournaments, and configuration options, see the [Tournaments](https://github.com/lowrollr/turbozero/wiki/Tournaments) wiki page.

### Demo
Demo mode provides the option to step through a game alongside an algorithm, which can be useful as a debugging tool or simply interesting to watch. For multi-player games, demo mode allows you to play *against* an algorithm, whether it be a heuristic baseline or a trained policy. For more information, see the [Demo](https://github.com/lowrollr/turbozero/wiki/Demo) page.

## Quickstart
### Google Colab
I've included a *Hello World* Google Colab notebook that runs through all of the main features of TurboZero and lets the user train and play against their own *Othello* AlphaZero model in only a few hours: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lowrollr/turbozero/blob/main/notebooks/hello_world_colab.ipynb)

If you'd rather run TurboZero on your own machine, follow the setup instructions below.

### Setup
The following commands will install poetry (dependency managaement), clone the repository, install required packages, and create a kernel for notebooks to connect to.

```terminal
curl -sSL https://install.python-poetry.org | python3 - && export PATH="/root/.local/bin:$PATH" && git clone https://github.com/lowrollr/turbozero.git && cd turbozero && poetry install && poetry run python -m ipykernel install --user --name turbozero
```
This will allow you to have access to the proper dependencies in Jupyter Notebooks by connecting to the `turbozero` kernel.

You can run scripts on the command-line by creating a shell using 
```terminal
poetry shell
```
If you'd rather not use poetry's shell, you can instead prepend `poetry run` to any commands.

### Training
To get started training a simple model, you can use one of the following commands, which load example configurations I've included for demonstration purposes. These commands will train a model and run periodic evaluation steps to track progress.

##### AlphaZero for Othello (CPU)
```terminal
python turbozero.py --verbose --mode=train --config=./example_configs/othello_tiny.yaml --logfile=./othello_tiny.log 
```
##### LazyZero for 2048 (CPU)
```terminal
python turbozero.py --verbose --mode=train --config=./example_configs/2048_tiny.yaml --logfile=./2048_tiny.log
```
The configuration files I've included train very small models and do not run many environments in parallel. You should be able to run this on your personal machine, but these commands will not train performant models.

If you have access to a GPU with CUDA, you can use the following commands to train slightly larger models. 
##### AlphaZero for Othello (GPU)
```terminal
python turbozero.py --verbose --gpu --mode=train --config=./example_configs/2048_mini.yaml --logfile=./othello_mini.log
```
##### LazyZero for 2048 (GPU)
```terminal
python turbozero.py --verbose --gpu --mode=train --config=./example_configs/2048_tiny.yaml --logfile=./2048_mini.log
```
With proper hardware these should not take long to train, as they are still relatively small. These commands will train on 4096 environments in parallel as opposed to 32 for the CPU configuration.

For more information on training configuration, please see the [Training](https://github.com/lowrollr/turbozero/wiki/Training) wiki page.

### Evaluation
If you'd like to evaluate an existing model, you can use `--mode=test`, link a checkpoint file with `--checkpoint`. For example:
```terminal
python turbozero.py --verbose --mode=test --config=./example_config/my_test_config.yaml --checkpoint=./checkpoints/my_checkpoint.pt --logfile=./test.log
```

For more information on evaluation/testing coniguration, see the [Evaluation & Testing](https://github.com/lowrollr/turbozero/wiki/Evaluation-&-Testing) wiki page.
### Tournament

To run an example tournament with some heuristic algorithms, you can run the following command:
```terminal
python turbozero.py --mode=tournament --config=./example_configs/othello_tournament.yaml
```

Remember to use the --gpu flag here if you have one, all algorithms are hardware accelerated!

For more information on tournament coniguration, see the [Tournaments](https://github.com/lowrollr/turbozero/wiki/Tournaments) wiki page.
### Demo
```terminal
python turbozero.py --mode=demo --config=./example_configs/othello_demo.yaml
```
For more information on demo coniguration, see the [Demo](https://github.com/lowrollr/turbozero/wiki/Demo) wiki page.
## Future Work
Major future initiatives include:
* implementing AZ improvements from [Clausen et al., 2021](https://www.scitepress.org/Papers/2021/102459/102459.pdf) as optional configurations
* Stochastic AlphaZero
* MuZero
* Investigate integration with 3rd-party vectorized environments
* Multi-GPU/Distributed support
* Additional vectorized environments
* Support for automated hyperparameter tuning

## Issues
If you use this project and encounter an issue, error, or undesired behavior, please submit a [GitHub Issue](https://github.com/lowrollr/turbozero/issues) and I will do my best to resolve it as soon as I can. You may also contact me directly via `hello@jacob.land`.

## Contributing 
Contributions, improvements, and fixes are more than welcome! I've written a lot in the [Wiki](https://github.com/lowrollr/turbozero/wiki), I hope it provides enough information to get started. For now I don't have a formal process for this, other than creating a [Pull Request](https://github.com/lowrollr/turbozero/pulls).

## Cite This Work
If you found this work useful, please cite it with:
```
@software{Marshall_TurboZero_Vectorized_AlphaZero,
  author = {Marshall, Jacob},
  title = {{TurboZero: Vectorized AlphaZero, MCTS, and Environments}},
  url = {https://github.com/lowrollr/turbozero}
}
```
