
from dataclasses import dataclass
import logging
from subprocess import PIPE, STDOUT, Popen
from typing import Optional
from core.algorithms.evaluator import Evaluator, EvaluatorConfig
from core.env import Env
from envs.othello.env import OthelloEnv
import torch

# this is an adapted version of EdaxPlayer: https://github.com/2Bear/othello-zero/blob/master/api.py

@dataclass
class EdaxConfig(EvaluatorConfig):
    edax_path: str
    edax_weights_path: str
    level: int



class Edax(Evaluator):
    def __init__(self, env: Env, config: EdaxConfig, *args, **kwargs):
        assert isinstance(env, OthelloEnv), "EdaxPlayer only supports OthelloEnv"
        super().__init__(env, config, *args, **kwargs)
        self.edax_exec = config.edax_path + " -q -eval-file " + config.edax_weights_path \
            + " -level " + str(config.level) \
            + " -book-randomness 10" \
            + " -n 4" 
            # + " -auto-store on"
        self.start_procs()
        self.read_stdout()
        
        

    def start_procs(self):
        self.edax_procs = [
            Popen(self.edax_exec, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT) for i in range(self.env.parallel_envs)
        ]

    def reset(self, seed=None):
        super().reset(seed)
        for proc in self.edax_procs:
            proc.terminate()
        self.start_procs()
        self.read_stdout()

    def step_evaluator(self, actions, terminated):
        # push other evaluators actions to edax
        for i, action in enumerate(actions.tolist()):
            if action == 64:
                self.write_to_proc("PS", i)
            else:
                self.write_to_proc(self.action_id_to_coord(action), i)
        self.read_stdout()
        self.reset_terminated_envs(terminated)
    
    def reset_terminated_envs(self, terminated) -> None:
        for i, t in enumerate(terminated):
            if t:
                self.edax_procs[i].terminate()
                self.edax_procs[i] = Popen(self.edax_exec, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
                self.read_stdout(i)
    
    def evaluate(self, *args):
        for i in range(len(self.edax_procs)):
            self.write_to_proc("go", i)
        results = self.read_stdout()
        actions = []
        for i, result in enumerate(results):
            if result != '\n\n*** Game Over ***\n':    
                move = self.coord_to_action_id(result.split("plays ")[-1][:2])
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
    
    @staticmethod
    def coord_to_action_id(coord):
        if coord == "PS":
            return 64
        letter = ord(coord[0]) - ord('A')
        number = int(coord[1]) - 1
        return letter + number * 8

    def write_stdin(self, command):
        self.edax.stdin.write(str.encode(command + "\n"))
        self.edax.stdin.flush()

    def read_stdout(self, proc_id: Optional[int] = None):
        outputs = []
        for i in range(len(self.edax_procs)):
            
            out = b''
            if proc_id is None or proc_id == i:
                while True:
                    next_b = self.edax_procs[i].stdout.read(1)
                    if next_b == b'>' and ((len(out) > 0 and out[-1] == 10) or len(out) == 0):
                        break
                    else:
                        out += next_b
            outputs.append(out)
        return [o.decode("utf-8") for o in outputs]

    def close(self):
        for p in self.edax_procs:
            p.terminate()
