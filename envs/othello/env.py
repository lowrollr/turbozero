
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
from colorama import Fore, Back, Style

from core.env import Env, EnvConfig
from .torchscripts import get_legal_actions, push_actions, build_filters, build_flips


@dataclass
class OthelloEnvConfig(EnvConfig):
    board_size: int = 8
    book: Optional[str] = None

def coord_to_action_id(coord):
    coord = coord.upper()
    if coord == "PS":
        return 64
    letter = ord(coord[0]) - ord('A')
    number = int(coord[1]) - 1
    return letter + number * 8


class OthelloEnv(Env):
    def __init__(self, 
        parallel_envs: int,
        config: OthelloEnvConfig,
        device: torch.device,
        debug=False
    ) -> None:
        self.board_size = config.board_size
        self.config: OthelloEnvConfig
        state_shape = torch.Size((2, self.board_size, self.board_size))
        policy_shape = torch.Size(((self.board_size ** 2) + 1,))
        value_shape = torch.Size((2, ))

        super().__init__(
            parallel_envs = parallel_envs,
            config = config,
            device=device,
            num_players=2, 
            state_shape=state_shape, 
            policy_shape=policy_shape, 
            value_shape=value_shape, 
            debug=debug
        )

        

        num_rays = (8 * (self.board_size - 2)) + 1
        self.ray_tensor = torch.zeros((self.parallel_envs, num_rays, self.board_size, self.board_size), dtype=torch.float32, device=device, requires_grad=False)
        
        self.filters_and_indices = build_filters(device, self.board_size)
        self.flips = build_flips(num_rays, self.board_size, device)

        self.consecutive_passes = torch.zeros((self.parallel_envs, ), dtype=torch.long, device=device, requires_grad=False)
        self.legal_actions = torch.zeros((self.parallel_envs, (self.board_size ** 2) + 1), dtype=torch.bool, device=device, requires_grad=False)

        self.need_to_calculate_rays = True
        

        if self.debug:
            self.get_legal_actions_traced = get_legal_actions
            self.push_actions_traced = push_actions
        else:
            self.get_legal_actions_traced = torch.jit.trace(get_legal_actions, (self.states, self.ray_tensor, self.legal_actions, *self.filters_and_indices)) # type: ignore
            self.push_actions_traced = torch.jit.trace(push_actions, (self.states, self.ray_tensor, torch.zeros((self.parallel_envs, ), dtype=torch.long, device=device), self.flips)) # type: ignore

        self.book_opening_actions = None
        if config.book is not None:
            self.book_opening_actions = self.parse_opening_book(config.book)

        self.reset()


    def parse_opening_book(self, path_to_book):
        with open(path_to_book, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines if l.strip() != '']
        ind = 0
        book_actions = []
        while ind < len(lines[0]):

            actions = torch.tensor([coord_to_action_id(line[ind:ind+2]) for line in lines], dtype=torch.long, device=self.device)
            book_actions.append(actions)
            ind += 2
        return torch.stack(book_actions)
                

    def get_legal_actions(self):
        if self.need_to_calculate_rays:
            self.need_to_calculate_rays = False
            return self.get_legal_actions_traced(self.states, self.ray_tensor, self.legal_actions, *self.filters_and_indices) # type: ignore
        else:
            return self.legal_actions
    
    def push_actions(self, actions):
        if self.need_to_calculate_rays:
            self.get_legal_actions() # updates ray tensor
        _, passes = self.push_actions_traced(self.states, self.ray_tensor, actions, self.flips) # type: ignore
        self.consecutive_passes += passes
        self.consecutive_passes *= passes
        self.need_to_calculate_rays = True
    
    def next_turn(self):
        self.states = torch.roll(self.states, 1, dims=1)
        self.next_player()
    
    def reset(self, seed=None) -> int:
        if seed is not None:
            torch.manual_seed(seed)
        seed = 0
        self.states.zero_()
        self.ray_tensor.zero_()
        self.terminated.zero_()
        self.cur_players.zero_()
        self.consecutive_passes.zero_()
        self.legal_actions.zero_()
        self.states[:, 0, 3, 4] = 1
        self.states[:, 1, 3, 3] = 1
        self.states[:, 1, 4, 4] = 1
        self.states[:, 0, 4, 3] = 1
        self.need_to_calculate_rays = True
        if self.book_opening_actions is not None:
            opening_ids = torch.randint(0, self.book_opening_actions.shape[1], (self.parallel_envs, ))
            for i in range(self.book_opening_actions.shape[0]):
                self.step(self.book_opening_actions[i, opening_ids])
        return seed


    def is_terminal(self):
        return (self.states.sum(dim=(1, 2, 3)) == (self.board_size ** 2)) | (self.consecutive_passes >= 2)
    
    def update_terminated(self):
        super().update_terminated()
    
    def get_rewards(self, player_ids: Optional[torch.Tensor] = None):
        if player_ids is None:
            player_ids = self.cur_players
        idx = ((player_ids == self.cur_players).int() - 1) % 2
        other_idx = 1 - idx

        p1_sum = self.states[self.env_indices, idx].sum(dim=(1, 2))
        p2_sum = self.states[self.env_indices, other_idx].sum(dim=(1, 2))
        rewards = (1 * (p1_sum > p2_sum)) + (0.5 * (p1_sum == p2_sum))
        return rewards

    def reset_terminated_states(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            torch.manual_seed(seed)
        seed = 0
        terminated = self.terminated.clone()
        self.states *= 1 * ~terminated.view(-1, 1, 1, 1)
        self.cur_players *= 1 * ~terminated
        self.consecutive_passes *= 1 * ~terminated
        mask = 1 * terminated
        self.states[:, 0, 3, 4] += mask
        self.states[:, 1, 3, 3] += mask
        self.states[:, 1, 4, 4] += mask
        self.states[:, 0, 4, 3] += mask
        
        saved = self.save_node()
        if self.book_opening_actions is not None:
            opening_ids = torch.randint(0, self.book_opening_actions.shape[1], (self.parallel_envs, ))
            for i in range(self.book_opening_actions.shape[0]):
                actions = self.book_opening_actions[i, opening_ids]
                actions[~terminated] = 64
                self.step(self.book_opening_actions[i, opening_ids])
        self.load_node(~terminated, saved)
        self.need_to_calculate_rays = True
        self.terminated.zero_()
        return seed

    def save_node(self):
        return (
            self.states.clone(),
            self.cur_players.clone(),
            self.consecutive_passes.clone()
        )
        
    def load_node(self, load_envs: torch.Tensor, saved: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        load_envs_expnd = load_envs.view(-1, 1, 1, 1)
        self.states = saved[0].clone() * load_envs_expnd + self.states * (~load_envs_expnd)
        self.cur_players = saved[1].clone() * load_envs + self.cur_players * (~load_envs)
        self.consecutive_passes = saved[2].clone() * load_envs + self.consecutive_passes * (~load_envs)
        self.need_to_calculate_rays = True
        self.update_terminated()
    
    def get_greedy_rewards(self, player_ids: Optional[torch.Tensor] = None, heuristic: str = 'num_tiles'):
        if heuristic == 'minmax_moves':
            legal_actions_sum = self.get_legal_actions().sum(dim=1)
            return (((legal_actions_sum) * (self.cur_players == player_ids)) + (self.policy_shape[0] - legal_actions_sum) * (self.cur_players != player_ids)) / self.policy_shape[0]
        elif heuristic == 'num_tiles':
            if player_ids is None:
                player_ids = self.cur_players
            idx = ((player_ids == self.cur_players).int() - 1) % 2
            other_idx = 1 - idx 
            return 0.5 + ((self.states[self.env_indices, idx].sum(dim=(1, 2)) - self.states[self.env_indices, other_idx].sum(dim=(1, 2))) / (2 * (self.board_size ** 2)))
        elif heuristic == 'corners':
            if player_ids is None:
                player_ids = self.cur_players
            idx = ((player_ids == self.cur_players).int() - 1) % 2
            other_idx = 1 - idx 
            top_left_corner = self.states[self.env_indices, idx, 0, 0] - self.states[self.env_indices, other_idx, 0, 0]
            top_right_corner = self.states[self.env_indices, idx, 0, self.board_size - 1] - self.states[self.env_indices, other_idx, 0, self.board_size - 1]
            bottom_left_corner = self.states[self.env_indices, idx, self.board_size - 1, 0] - self.states[self.env_indices, other_idx, self.board_size - 1, 0]
            bottom_right_corner = self.states[self.env_indices, idx, self.board_size - 1, self.board_size - 1] - self.states[self.env_indices, other_idx, self.board_size - 1, self.board_size - 1]
            return 0.5 + ((top_left_corner + top_right_corner + bottom_left_corner + bottom_right_corner) / 8)
        elif heuristic == 'corners_and_edges':
            if player_ids is None:
                player_ids = self.cur_players
            idx = ((player_ids == self.cur_players).int() - 1) % 2
            other_idx = 1 - idx
            edge = self.states[self.env_indices, idx, 0, :] - self.states[self.env_indices, other_idx, 0, :]
            edge += self.states[self.env_indices, idx, :, 0] - self.states[self.env_indices, other_idx, :, 0]
            edge += self.states[self.env_indices, idx, :, self.board_size - 1] - self.states[self.env_indices, other_idx, :, self.board_size - 1]
            edge += self.states[self.env_indices, idx, self.board_size - 1, :] - self.states[self.env_indices, other_idx, self.board_size - 1, :]
            # corners are counted twice, but corners are good, so it's fine!
            circumference = (2 * self.board_size) + (2 * (self.board_size - 2))
            return 0.5 + (edge / (2 * circumference))

        else:
            raise NotImplementedError(f'Heuristic {heuristic} not implemented for OthelloEnv')

    def print_state(self, last_action: Optional[int] = None) -> None:
        envstr = []
        assert self.parallel_envs == 1
        cur_player_is_o = self.cur_players[0] == 0
        cur_player = 'O' if cur_player_is_o else 'X'
        other_player = 'X' if cur_player_is_o else 'O'
        envstr.append('+' + '---+' * (self.config.board_size))
        envstr.append('\n')
        legal_actions = set(self.get_legal_actions()[0].nonzero().flatten().tolist())
        for i in range(self.config.board_size):
            for j in range(self.config.board_size):
                action_idx = i*self.config.board_size + j
                color = Fore.RED if cur_player_is_o else Fore.GREEN
                other_color = Fore.GREEN if cur_player_is_o else Fore.RED
                if action_idx == last_action:
                    color = Fore.BLUE
                    other_color = Fore.BLUE
                if action_idx in legal_actions: 
                    envstr.append('|' + Fore.YELLOW + f'{action_idx}'.rjust(3))
                elif self.states[0,0,i,j] == 1:
                    envstr.append('|' + color + f' {cur_player} '.rjust(3))
                elif self.states[0,1,i,j] == 1:
                    envstr.append('|' + other_color + f' {other_player} '.rjust(3))
                else:
                    envstr.append(Fore.RESET + '|   ')
                envstr.append(Fore.RESET)
            envstr.append('|')
            envstr.append('\n')
            envstr.append('+' + '---+' * (self.config.board_size))
            envstr.append('\n')
        print(''.join(envstr))
