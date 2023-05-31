import torch


class VectorizedTrainer:
    def __init__(self, evaluator, model, iters, depth) -> None:
        self.eval = evaluator
        self.model = model
        self.lmcts_iters = iters
        self.lmcts_depth = depth
        self.unfinished_games = [[] for _ in range(self.eval.env.num_parallel_envs)]
        self.memory_buffer = []

    def train(self):
        pass

    def push_examples_to_memory_buffer(self, terminated):
        terminated_envs = torch.nonzero(terminated.view(self.eval.env.num_parallel_envs)).cpu().numpy()
        if len(terminated_envs):
            high_squares = torch.amax(self.eval.env.boards, dim=(1,2,3)).cpu().numpy()
            for t in terminated_envs:
                ind = t[0]
                moves = len(self.unfinished_games[ind])
                high_square = high_squares[ind]
                self.memory_buffer.append((self.unfinished_games[ind], moves, high_square))
                self.unfinished_games[ind] = []
                


    def run_step(self):
        self.model.eval()
        visits = self.eval.explore(self.lmcts_iters, self.lmcts_depth)
        np_boards = self.eval.env.boards.clone().cpu().numpy()
        actions = torch.argmax(visits, dim=1)
        terminated = self.eval.env.step(actions)
        np_visits = visits.cpu().numpy()
        for i in range(self.eval.env.num_parallel_envs):
            self.unfinished_games[i].append((np_boards[i], np_visits[i]))
        self.push_examples_to_memory_buffer(terminated)
        self.eval.env.reset_invalid_boards()
        return np_boards, actions, terminated