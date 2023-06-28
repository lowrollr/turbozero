




import torch
from core.memory import EpisodeMemory
from .lazy_mcts import VectorizedLazyMCTS


class Collector:
    def __init__(self, 
        evaluator: VectorizedLazyMCTS, 
        episode_memory_device: torch.device, 
        search_iters: int, 
        search_depth: int
    ) -> None:
        num_parallel_envs = evaluator.env.num_parallel_envs
        self.evaluator = evaluator
        self.evaluator.reset()
        self.episode_memory = EpisodeMemory(num_parallel_envs, episode_memory_device)
        self.search_iters = search_iters
        self.search_depth = search_depth
        self.epsilon_sampler = torch.zeros((num_parallel_envs,), dtype=torch.float32, device=evaluator.env.device, requires_grad=False)

    def collect(self, model: torch.nn.Module, epsilon: float = 0.0, reset_terminal: bool = True):
        model.eval()

        visits = self.evaluator.explore(model, self.search_iters, self.search_depth)

        self.episode_memory.insert(self.evaluator.env.states, visits)

        torch.rand(self.epsilon_sampler.shape, out=self.epsilon_sampler)
        take_argmax = self.epsilon_sampler > epsilon
        actions = torch.argmax(visits, dim=1) * take_argmax
        actions += self.evaluator.env.fast_weighted_sample(visits) * (~take_argmax)

        terminated, info = self.evaluator.env.step(actions)

        terminated_episodes = self.episode_memory.pop_terminated_episodes(terminated)

        terminated_episodes = self.assign_rewards(terminated_episodes, terminated, info)

        if reset_terminal:
            self.evaluator.env.reset_terminated_states()

        return terminated_episodes, terminated
    
    def assign_rewards(self, terminated_episodes, terminated, info):
        raise NotImplementedError()
    
    
    def postprocess(self, terminated_episodes):
        return terminated_episodes
    
    def reset(self):
        self.episode_memory = EpisodeMemory(self.evaluator.env.num_parallel_envs, self.episode_memory.device)
        self.evaluator.reset()
