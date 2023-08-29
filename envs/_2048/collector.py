



from typing import Optional
import torch
from core.train.collector import Collector
from core.algorithms.evaluator import Evaluator

class _2048Collector(Collector):
    def __init__(self,
        evaluator: Evaluator,
        episode_memory_device: torch.device
    ) -> None:
        super().__init__(evaluator, episode_memory_device)

    def assign_rewards(self, terminated_episodes, terminated):
        episodes = []
        for episode in terminated_episodes:
            episode_with_rewards = []
            moves = len(episode)
            for (inputs, visits, legal_actions) in episode:
                episode_with_rewards.append((inputs, visits, torch.tensor(moves, dtype=torch.float32, requires_grad=False, device=inputs.device), legal_actions))
                moves -= 1
            episodes.append(episode_with_rewards)
        return episodes

    def postprocess(self, terminated_episodes):
        # TODO: too many lists
        inputs, probs, rewards, legal_actions = zip(*terminated_episodes)
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
        rotated_legal_actions = []
        for l in legal_actions:
            for k in range(4):
                rotated_legal_actions.append(torch.roll(l, k))
        rotated_rewards = []
        for r in rewards:
            rotated_rewards.extend([r] * 4)
        
        return list(zip(rotated_inputs, rotated_probs, rotated_rewards, rotated_legal_actions))
            