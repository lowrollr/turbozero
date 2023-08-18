

from core.demo.demo import TwoPlayerDemo


class OthelloDemo(TwoPlayerDemo):


    def print_rewards(self, p1_started: bool):
        super().print_rewards(p1_started)
        p1_idx = 0 if self.evaluator.env.cur_players.item() == 0 else 1
        p2_idx = 1 if p1_idx == 0 else 0
        p1_tiles = self.evaluator.env.states[0,p1_idx].sum().item()
        p2_tiles = self.evaluator.env.states[0,p2_idx].sum().item()
        print(f'Player 1 Tiles: {int(p1_tiles)}')
        print(f'Player 2 Tiles: {int(p2_tiles)}')