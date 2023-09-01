
from dataclasses import dataclass
import logging
from subprocess import PIPE, STDOUT, Popen
from typing import Optional, Set
from core.algorithms.evaluator import Evaluator, EvaluatorConfig
from core.env import Env
from envs.othello.env import OthelloEnv, coord_to_action_id
import torch

# this is an adapted version of EdaxPlayer: https://github.com/2Bear/othello-zero/blob/master/api.py

@dataclass
class EdaxConfig(EvaluatorConfig):
    edax_path: str
    edax_weights_path: str
    edax_book_path: str
    level: int



class Edax(Evaluator):
    def __init__(self, env: Env, config: EdaxConfig, *args, **kwargs):
        assert isinstance(env, OthelloEnv), "EdaxPlayer only supports OthelloEnv"
        super().__init__(env, config, *args, **kwargs)
        self.edax_exec = config.edax_path + " -q -eval-file " + config.edax_weights_path \
            + " -level " + str(config.level) \
            + " -n 4"
        self.edax_procs = []
        

    def start_procs(self):
        self.edax_procs = [
            Popen(self.edax_exec, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT) for i in range(self.env.parallel_envs)
        ]

    def reset(self, seed: Optional[int] = None) -> int:
        seed = super().reset(seed)
        for proc in self.edax_procs:
            proc.terminate()
        self.start_procs()
        self.read_stdout()
        for i in range(len(self.edax_procs)):
            self.write_to_proc(f"setboard {self.stringify_board(i)}", i)
        self.read_stdout()
        return seed
    
    def stringify_board(self, index):
        cur_player = self.env.cur_players[index]
        board = self.env.states[index]
        black_pieces = board[0] if cur_player == 0 else board[1]
        white_pieces = board[1] if cur_player == 0 else board[0]
        board_str = []
        for i in range(64):
            r, c = i // 8, i % 8
            if black_pieces[r][c] == 1:
                board_str.append('b')
            elif white_pieces[r][c] == 1:
                board_str.append('w')
            else:
                board_str.append('-')
        return ''.join(board_str)


    def step_evaluator(self, actions, terminated):
        # push other evaluators actions to edax
        for i, action in enumerate(actions.tolist()):
            if action == 64:
                self.write_to_proc("PS", i)
            else:
                self.write_to_proc(self.action_id_to_coord(action), i)
        self.read_stdout()
        self.reset_evaluator_states(terminated)
    
    def reset_evaluator_states(self, evals_to_reset: torch.Tensor) -> None:
        read_from_procs = set()
        for i, t in enumerate(evals_to_reset):
            if t:
                self.edax_procs[i].terminate()
                self.edax_procs[i] = Popen(self.edax_exec, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
                read_from_procs.add(i)
        if read_from_procs:
            self.read_stdout(read_from_procs)
            for i in read_from_procs:
                self.write_to_proc(f"setboard {self.stringify_board(i)}", i)
            self.read_stdout(read_from_procs)
    
    def evaluate(self, *args):
        for i in range(len(self.edax_procs)):
            self.write_to_proc("go", i)
        results = self.read_stdout()
        actions = []
        for i, result in enumerate(results):
            if result != '\n\n*** Game Over ***\n':    
                move = coord_to_action_id(result.split("plays ")[-1][:2])
                actions.append(move)
            else:
                # logging.info(f'PROCESS {i}: EDAX gives GAME OVER')
                actions.append(64)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        return torch.nn.functional.one_hot(actions, self.env.policy_shape[0]).float(), None
    
    def step_env(self, actions):
        terminated = self.env.step(actions)
        return terminated

    def write_to_proc(self, command, proc_id):
        self.edax_procs[proc_id].stdin.write(str.encode(command + "\n"))
        self.edax_procs[proc_id].stdin.flush()
    
    @staticmethod
    def action_id_to_coord(action_id):
        letter = chr(ord('a') + action_id % 8)
        number = str(action_id // 8 + 1)
        if action_id == 64:
            return "PS"
        
        return letter + number

    def write_stdin(self, command):
        self.edax.stdin.write(str.encode(command + "\n"))
        self.edax.stdin.flush()

    def read_stdout(self, proc_ids: Optional[Set[int]] = None):
        outputs = []
        for i in range(len(self.edax_procs)):
            out = b''
            if proc_ids is None or i in proc_ids:
                while True:
                    next_b = self.edax_procs[i].stdout.read(1)
                    if next_b == b'>' and ((len(out) > 0 and out[-1] == 10) or len(out) == 0):
                        break
                    else:
                        out += next_b
            outputs.append(out)
        decoded = [o.decode("utf-8") for o in outputs]
        return decoded

    def close(self):
        for p in self.edax_procs:
            p.terminate()
