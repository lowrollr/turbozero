




import torch
from core.utils.memory import EpisodeMemory
from core.evaluation.evaluator import Evaluator


class Collector:
    def __init__(self, 
        evaluator: Evaluator, 
        episode_memory_device: torch.device
    ) -> None:
        num_parallel_envs = evaluator.env.num_parallel_envs
        self.evaluator = evaluator
        self.evaluator.reset()
        self.episode_memory = EpisodeMemory(num_parallel_envs, episode_memory_device)
        self.epsilon_sampler = torch.zeros((num_parallel_envs,), dtype=torch.float32, device=evaluator.env.device, requires_grad=False)

    def collect(self, model: torch.nn.Module, epsilon: float = 0.0, reset_terminal: bool = True):
        
        terminated = self.collect_step(model, epsilon)

        terminated_episodes = self.episode_memory.pop_terminated_episodes(terminated)

        terminated_episodes = self.assign_rewards(terminated_episodes, terminated)

        if reset_terminal:
            self.evaluator.env.reset_terminated_states()

        return terminated_episodes, terminated
    
    def collect_step(self, model: torch.nn.Module, epsilon: float = 0.0):
        model.eval()

        visits = self.evaluator.evaluate(model)
        self.episode_memory.insert(self.evaluator.env.states, visits)

        torch.rand(self.epsilon_sampler.shape, out=self.epsilon_sampler)
        take_argmax = self.epsilon_sampler >= epsilon
        actions = torch.argmax(visits, dim=1) * take_argmax
        actions += self.evaluator.env.fast_weighted_sample(visits) * (~take_argmax)

        terminated = self.evaluator.step_env(actions)

        return terminated
    
    def assign_rewards(self, terminated_episodes, terminated):
        raise NotImplementedError()
    
    def postprocess(self, terminated_episodes):
        return terminated_episodes
    
    def reset(self):
        self.episode_memory = EpisodeMemory(self.evaluator.env.num_parallel_envs, self.episode_memory.device)
        self.evaluator.reset()

    def get_details(self):
        return {
            'type': type(self.evaluator),
            'hypers': self.evaluator.hypers,
        }
