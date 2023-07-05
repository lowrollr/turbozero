



import torch
from core.training.collector import Collector
from core.evaluation.evaluator import Evaluator
from .evaluator import _2048_EVALUATORS

class _2048Collector(Collector):
    def __init__(self,
        evaluator: _2048_EVALUATORS,
        episode_memory_device: torch.device
    ) -> None:
        super().__init__(evaluator, episode_memory_device)

    def assign_rewards(self, terminated_episodes, terminated):
        episodes = []
        for episode in terminated_episodes:
            episode_with_rewards = []
            moves = len(episode)
            for (inputs, visits) in episode:
                episode_with_rewards.append((inputs, visits, torch.tensor(moves, dtype=torch.float32, requires_grad=False, device=inputs.device)))
                moves -= 1
            episodes.append(episode_with_rewards)
        return episodes

    def postprocess(self, terminated_episodes):
        # TODO: too many lists
        inputs, probs, rewards = zip(*terminated_episodes)
        rotated_inputs = []
        for i in inputs:
            for k in range(4):
                rotated_inputs.append(torch.rot90(i, k=k, dims=(1, 2)))
        rotated_probs = []
        for p in probs:
            # left -> down
            # down -> right
            # right -> up
            # up -> left
            for k in range(4):
                rotated_probs.append(torch.roll(p, k))
        rotated_rewards = []
        for r in rewards:
            rotated_rewards.extend([r] * 4)
        
        return list(zip(rotated_inputs, rotated_probs, rotated_rewards))
            