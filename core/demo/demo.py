
import random
import torch
from core.algorithms.evaluator import Evaluator
from IPython.display import clear_output
import os

class Demo:
    def __init__(self,
        evaluator: Evaluator,
        manual_step: bool = False
    ):
        self.manual_step = manual_step
        self.evaluator = evaluator
        assert self.evaluator.env.parallel_envs == 1

    def run(self, print_evaluation: bool = False, print_state: bool = True, interactive: bool =True):
        self.evaluator.reset()
        actions = None
        while True:
            if print_state:
                self.evaluator.env.print_state(actions.item() if actions is not None else None)
            if self.manual_step:
                input('Press any key to continue...')
            _, _, value, actions, terminated = self.evaluator.step()
            if interactive:
                clear_output(wait=True)
            else:
                os.system('clear')
            if print_evaluation and value is not None:
                print(f'Evaluation: {value[0]}')
            if terminated:
                print('Game over!')
                print('Final state:')
                print(self.evaluator.env)
                break

class TwoPlayerDemo(Demo):
    def __init__(self,
        evaluator,
        evaluator2,
        manual_step: bool = False
    ) -> None:
        super().__init__(evaluator, manual_step)
        self.evaluator2 = evaluator2
        assert self.evaluator2.env.parallel_envs == 1
    
    def run(self, print_evaluation: bool = False, print_state: bool = True, interactive: bool =True):
        seed = random.randint(0, 2**32 - 1)
        
        self.evaluator.reset(seed)
        self.evaluator2.reset(seed)
        p1_turn = random.choice([True, False])
        cur_player = self.evaluator.env.cur_players.item()
        p1_player_id = cur_player if p1_turn else 1 - cur_player
        p1_evaluation = 0.5
        p2_evaluation = 0.5
        actions = None
        while True:
            

            active_evaluator = self.evaluator if p1_turn else self.evaluator2
            other_evaluator = self.evaluator2 if p1_turn else self.evaluator
            if print_state:
                print(f'Player 1 (O): {self.evaluator.__class__.__name__ if p1_player_id == 0 else self.evaluator2.__class__.__name__}')
                print(f'Player 2 (X): {self.evaluator.__class__.__name__ if p1_player_id == 1 else self.evaluator2.__class__.__name__}')
                active_evaluator.env.print_state(int(actions.item()) if actions is not None else None)
            if self.manual_step:
                input('Press any key to continue...')
            _, _, value, actions, terminated = active_evaluator.step()
            if p1_turn:
                p1_evaluation = value[0] if value is not None else None
            else:
                p2_evaluation = value[0] if value is not None else None
            other_evaluator.step_evaluator(actions, terminated)
            if interactive:
                clear_output(wait=False)
            else:
                os.system('clear')
            if print_evaluation:
                if p1_evaluation is not None:
                    print(f'{self.evaluator.__class__.__name__} Evaluation: {p1_evaluation}')
                if p2_evaluation is not None:
                    print(f'{self.evaluator2.__class__.__name__} Evaluation: {p2_evaluation}')
            if terminated:
                print('Game over!')
                print('Final state:')
                active_evaluator.env.print_state(int(actions.item()))
                self.print_rewards(p1_player_id)
                break
                
            p1_turn = not p1_turn

    def print_rewards(self, p1_player_id):
        reward = self.evaluator.env.get_rewards(torch.tensor([p1_player_id]))[0]
        if reward == 1:
            print(f'Player 1 ({self.evaluator.__class__.__name__}) won!')
        elif reward == 0:
            print(f'Player 2 ({self.evaluator2.__class__.__name__}) won!')
        else:
            print('Draw!')