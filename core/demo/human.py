


import torch
from core.algorithms.evaluator import Evaluator


class HumanEvaluator(Evaluator):
    def __init__(self, env, config):
        super().__init__(env, config)
        assert self.env.parallel_envs == 1, 'HumanEvaluator only supports parallel_envs=1'

    def evaluate(self):
        legal_action_ids = []
        for index, i in enumerate(self.env.get_legal_actions()[0]):
            if i:
                legal_action_ids.append(index)
        
        print('Legal actions:', legal_action_ids)
        
        while True:
            action = input('Enter action: ')
            try:
                action = int(action)
                if action in legal_action_ids:
                    break
                else:
                    print('Action not legal, choose a legal action')
            except:
                print('Invalid input')

        return torch.nn.functional.one_hot(torch.tensor([action]), num_classes=self.env.policy_shape[0]).float(), None

