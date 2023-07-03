# LazyZero

-- You can pretty much ignore everything you read below this, I'll be re-writing the readme (and re-naming the project) soon. I figured out how to vectorize non-naive AlphaZero and will be working on that going forward, expect updates soon -- 

LazyZero is an approximate, GPU-accelerated, vectorized implementation of DeepMind's AlphaZero reinforcement learning algorithm. AlphaZero is an algorithm that utilizes Deep Neural Networks alongside Monte Carlo Tree Search to make decisions in perfect-information games, which has since been extended to other applications, such as video compression and code generation. 

I speak a lot about AlphaZero and Monte Carlo Tree Search (MCTS) in this wiki, it would be useful to read the following atricle before jumping in if you aren't familiar with these concepts: https://web.stanford.edu/~surag/posts/alphazero.html. This article does a great job explaining the intuition behind AlphaZero and illustrating concepts with code.

One issue with AlphaZero is that it is incredibly compute-inefficient to train, oftentimes requiring hundreds of model inference calls and even more game logic calls per episode step. AlphaGo, for instance, could only be trained with thousands of TPUs. This costliness combined with the need to train on millions of episodes means that as problem complexity scales, compute requirements may scale to the point where it becomes infeasible to effectively train a model.

While efforts have been made to better parallelize AlphaZero using batch processing, there (to my knowledge) is a lack of a true vectorized approch that takes full advantage of GPU-parallelism. 

The efficiency issue becomes less of a problem in production settings. Once a model is trained and deployed, the use-case is often real-time, one-off decisions (i.e., making a move in a chess game) where parallelization matters much less. To more specifically define the issue: AlphaZero has a *training efficiency* problem.

In this project I introduce __*LazyZero*__, which solves the training efficiency problem by utilizing a vectorized MCTS-esque algorithm, *LazyMCTS*, that dramatically increases training performance at the cost of lower model accuracy. The learned policy and value-estimation are transferrable, such that they can be used with a non-approximate MCTS algorithm in production settings. 

The LazyZero project aims to introduce a framework for fully-vectorized training of AlphaZero-esque models with the following features:
* __end-to-end GPU acceleration:__ environment logic/simulation + model training/inference all happens without data ever leaving the GPU
    * __WIP:__ the current implementation stores training examples in the Replay Memory buffer in RAM rather than on the GPU, in order to truly be GPU-accelerated end-to-end this mechanism will be moved to the GPU as well
* __a lazy, vectorized implementation of MCTS:__ rather than utilize the resource-hungry full implementation of MCTS, we utilize a lazy version that identifies branches to explore during each iteration using PUCT at the root node, then samples from the learned policy to carry out fixed-depth rollouts. This approach is completely vectorizable and can be carried out across thousands of environments at once. 
    * While this approach ultimately only trains an approximation of a true-alphazero policy, it is many magnitudes more efficient to train, and I hope to prove it yields a *close* approximation of the non-lazy policy. In compute-constrained environments, LazyZero could provide a viable alternative where AlphaZero is infeasible.
    * *__Key Idea: The learned policy is algorithm-agnostic, which means production environments could still utilize full-MCTS while using the lazily-trained policy__*
* __a gymnasium-esque framework for defining vectorized environments:__ we define base classes for vectorized environments that closely resemble the gymnasium environment spec. 
    * we currently implement an environment for stochastic, single-player games. Future work will implement the traditional multi-agent environment as well
    * Future work may also include a migration to be gymnasium-compatible
* __toy examples__: Train a model with LazyZero to play and win *2048*. Play hundreds of thousands of games in parallel on a single GPU! I'm working on expanding this to multi-player games as well.

While LazyZero shows promise, it is still just a hypothesis rather than a proven framework. While environments like *2048* are vectorizable, it is unclear whether that holds true across more sophisticated, interesting use-cases. In addition, further testing and research is necessary to determine whether or not LazyZero provides a good-enough approximation of true AlphaZero to merit use. *There is much work to be done to answer these questions.*

If you'd like to collaborate on these ideas with me, please reach out! 

## Implementation Details
### VectEnv
In order to take advantage of GPU parallelism, environments must be implemented as a stacked set of states that operations can be applied to in parallel. In the case of *2048* this looks something like this:
![image](./misc/2048env.png)

This representation allows us to perform model inference on many environments at once, which is a big improvement on other open-source implementations of AlphaZero. But perhaps of even greater significance, this representation allows us to vectorize operations on environment states. LazyZero follows a very simple workflow while collecting episodes during training:
![image](./misc/workflow.png)

We use a similar workflow within LazyMCTS to explore subsequent game states.

In AlphaZero implementations, you usually see something more akin to this:
![image](./misc/cpu_bad.png)

While many implementations mitigate this obvious downside with cpu-based parallelism (multi-processing) alongside clever batching, these methods fail to capture the dramatic performance upside of doing everything on the GPU.

A key drawback of LazyMCTS is that environments *must* be vectorized. If implementation details or charistertics of an environment make it impossible or inefficient to vectorize, LazyMCTS will not provide any meaningful throughput improvement over CPU-based parallelism. A nice property of *2048* is that all necessary computation can be written as a set of matrix operations, which means that we can vectorize these operations and apply them to every state at once. I'll illustrate with a few examples:

In 2048, the possible actions in any given state are sliding Up, Down, Left, and Right. A slide action is legal if any tile can slide in the given direction, or if two similar tiles slide into one another and combine. We can use the following operations to check the legality of each action for a given state. In this example (and in the environment I implemented), I use powers of 2 (e.g. 8-tile = 3, 2048-tile = 11) to represent the game tiles. I use '0' to denote an empty grid space.

![image](./misc/slide_conv.png)

We also need to check for the case where identical-tiles merge, for this example I'll just show the code:

```python
# Compute the differences between adjacent elements in the 2nd and 3rd dimensions
vertical_diff = states[:, :, :-1, :] - states[:, :, 1:, :]
horizontal_diff = states[:, :, :, :-1] - states[:, :, :, 1:]

# Check where the differences are zero, excluding the zero elements in the original matrix
vertical_zeros = torch.logical_and(vertical_diff == 0, states[:, :, 1:, :] != 0)
horizontal_zeros = torch.logical_and(horizontal_diff == 0, states[:, :, :, 1:] != 0)
```

Notice that in this example code we are operating on 4 dimensions, because these are vectorized operations across many 1x4x4 environments. With some help from our GPU we can simulataneously compute the legal actions across N boards!

All LazyZero environments inherit from the VectEnv base class. Much like gymnasium, this class implements step(action) and reset():

However, certain details make this implementation incompatible with other aspects of gymnasium, so for the time being these environments are not compatible with gymnasium tooling.

### LazyMCTS
LazyMCTS approximates true Monte Carlo Tree Search by performing MCTS-esque exploration/exploitation decisions at the root node only, and then carrying out fixed-depth rollouts sampling from the current trained policy. Maintining W and N values at the root node is trivial and easily vectorizable, but maintining these values for a dynamically expanding tree structure is infeasible to vectorize, so LazyMCTS performs fixed-depth rollouts sampling from the model policy instead.

![image](./misc/one_lazyzero_iteration.png)

With a vectorized environment and a vectorized search algorithm, we can perform an episode step using LazyMCTS across many environments all at once, without data ever leaving the GPU. We can take this further and perform entire training or evaluation episodes entirely on the GPU. 

This leads into the issue of storing training examples in the Replay Memory Buffer, a key component of AlphaZero. I do not currently have a GPU-based solution for this, as I've ran into use-cases with reward assignment that makes storing unfinished episode steps on the GPU non-trivial. I hope to provide an update where this data never leaves the GPU in the near future.

My MCTS implementation also borrows a technique from other reinforcement-learning algorithms: using an [epsilon-greedy policy](https://medium.com/analytics-vidhya/the-epsilon-greedy-algorithm-for-reinforcement-learning-5fe6f96dc870). This may work better in some environments and worse in others, but I acheived better results for *2048* when using an epsilon value decaying from 1.0 to 0.01, YMMV.



## Results
I've been using the toy environment *2048* to iterate and improve upon the vectorized implementation of MCTS. I'm planning on implementing more sophisticated environments for single-player and multi-player games very soon. 

I experimented with many different rewards structures for *2048*, but found the best results by assigning each state a reward equal to the number of actions taken during the episode after that state. (e.g. In a game with 100 moves, state 0 is assigned a reward of 100, state 1 is assigned a reward of 99, etc.). 

I did not focus any of my time on tuning hyperparameters but still managed to acheive some encouraging results: in evaluation settings the model managed to acheive a score of 2048 or higher in 96% of games, and scored 4096 or higher in over half!

![image](./misc/high_tile.png)

Here are a few statistics from the training run, which was comprised of 500,000 episodes and took roughly 3 days to run on an A100 GPU.

![image](./misc/train_high_tile.png)
![image](./misc/train_moves.png)
![image](./misc/train_accuracy.png)

*2048* mostly served as a very simple-to-implement (relatively) environment to validate that this training approach *could* be effective. I'm now working on extending this to multi-player environments to see if I can also acheive good results there. I also would like to gather results when utilizing the trained *2048* model with full-MCTS, in order to find out the extent of the accuracy loss for the approximate algorithm.

If you'd like to train the *2048* model yourself, you can do so in the *vectorized.ipynb* notebook. I intend to write some more documentation on the VectTrainer class and surrounding utilies at some point.


## Future/Ongoing Work
* support for multi-player environments
* train on multi-player environments (othello is next)
* cpu-parallelism for training in non-GPU computing environments using multiprocessing
* full-MCTS implementation for production environments
* GPU replay memory buffer
* more wiki content
