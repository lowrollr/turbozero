from .vectenv import VectEnv
import torch


class MPVectEnv(VectEnv):
    def __init__(self, num_players, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_player = torch.zeros(
            (self.num_parallel_envs,),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        self.num_players = num_players

    def step(self, actions):
        terminated, info = super().step(actions)
        self.cur_player = (self.cur_player + 1) % self.num_players
        info['rewards'] = self.get_rewards()
        self.next_turn()
        return terminated, info
    
    def next_turn(self):
        raise NotImplementedError()
    
    def get_rewards(self):
        raise NotImplementedError()
    
    
    
    
