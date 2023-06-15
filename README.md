# LazyZero

LazyZero is an approximate, GPU-accelerated, vectorized implementation of DeepMind's AlphaZero reinforcement learning algorithm. AlphaZero is an algorithm that utilizes Deep Neural Networks alongside Monte Carlo Tree Search to make decisions in perfect-information games, which has since been extended to other applications, such as video compression and code generation. 

AlphaZero is incredibly compute-inefficient, oftentimes requiring hundreds of thousands of model inference and game logic calls per episode step. AlphaGo, for instance, could only be trained with thousands of TPUs. While efforts have been made to better parallelize AlphaZero using batch processing, there (to my knowledge) is a lack of a true vectorized approch that takes full advantage of GPU-parallelism. 

The LazyZero project aims to introduce a framework for fully-vectorized training of AlphaZero agents with the following features:
* __end-to-end GPU acceleration:__ environment logic/simulation + model training/inference all happens without data ever leaving the GPU
    * __WIP:__ the current implementation stores training examples in the Replay Memory buffer in RAM rather than on the GPU, in order to truly be GPU-accelerated end-to-end this mechanism will be moved to the GPU as well
* __a lazy, vectorized implementation of MCTS:__ rather than utilize the resource-hungry full implementation of MCTS, we utilize a lazy version that identifies branches to explore during each iteration using PUCT at the root node, and then samples from a policy to carry out shallow rollouts. This approach is completely vectorized and can be carried out across thousands of environments at once. 
    * While this approach ultimately only trains an approximation of a true-alphazero policy, it is many magnitudes more efficient to train, and I hope to prove it yields a *close* approximation of the non-lazy policy. In compute-constrained environments, LazyZero could provide a viable alternative where AlphaZero is infeasible.
    * *__Key Idea: The learned policy is algorithm-agnostic, which means production environments could still utilize full-MCTS while using the lazily-trained policy__*
* __a gymnasium-esque framework for defining vectorized environments:__ we define base classes for vectorized environments that closely resemble the gymnasium environment spec. 
    * we currently implement an environment for stochastic, single-player games. Future work will implement the traditional multi-agent environment as well
    * Future work may also include a migration to be gymnasium-compatible
* __toy examples__: Train a model with LazyZero to play and win *2048*. Play hundreds of thousands of games in parallel on a single GPU!

While LazyZero shows promise, it is still just a hypothesis rather than a proven framework. While environments like *2048* are vectorizable, it is unclear whether that holds true across more sophisticated, interesting use-cases. In addition, further testing and research is necessary to determine whether or not LazyZero provides a good-enough approximation of true AlphaZero to merit use. *There is much work to be done to answer these questions.*

If you'd like to collaborate on these ideas with me, please reach out! 


## Planned Features
* support for multi-player environments
* cpu-parallelism for training in non-GPU computing environments using multiprocessing
* implement full-MCTS for production environments
* GPU replay memory buffer