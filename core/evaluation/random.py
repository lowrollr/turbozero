




import torch
from core.evaluation.evaluator import Evaluator
from core.vectenv import VectEnv


class RandomBaseline(Evaluator):
    def evaluate(self):
        legal_actions = self.env.get_legal_actions().float()
        return torch.multinomial(legal_actions, 1, replacement=True).flatten()