
import argparse
import torch
import random
from core.evaluation.mcts_hypers import MCTSHypers
from core.resnet import TurboZeroArchParams, TurboZeroResnet
from core.training.training_hypers import TurboZeroHypers
from envs.othello.evaluator import OthelloMCTS

from envs.othello.vectenv import OthelloVectEnv



def play(model: torch.nn.Module, env: OthelloVectEnv, evaluator: OthelloMCTS):
    playing_as_white = random.randint(0, 1)
    if playing_as_white:
        print('Playing as X')
    else:
        print('Playing as O')

    player_turn = not playing_as_white

    print('Starting game')
    env.reset()
    

    while not env.is_terminal()[0]:
        print_env(env)
        if player_turn:
            legal_actions = env.get_legal_actions()[0].nonzero().flatten().tolist()
            if legal_actions:
                print(
                    'Legal actions: ' + ', '.join([str(i) for i in legal_actions])
                )
                while True:
                    move = input('Enter move: ')
                    if int(move) in legal_actions:
                        break
                    else:
                        print('Invalid move')
                        
                env.step(torch.tensor([int(move)]))
        else:
            visits = evaluator.evaluate(model)[0]
            move = torch.argmax(visits)
            evaluator.step_env(torch.tensor([int(move)]))
        player_turn = not player_turn
    print_env(env)
    print('Game over!')
    r1, r2 = env.states.sum(dim=(2, 3))[0].tolist()
    if playing_as_white:
        player_score, opp_score = (r1, r2) if env.cur_players[0].item() else (r2, r1)
    else:
        player_score, opp_score = (r2, r1) if env.cur_players[0].item() else (r1, r2)
    if player_score > opp_score:
        print('You won!')
    elif player_score < opp_score:
        print('You lost!')
    elif player_score == opp_score:
        print('Draw!')
    print(f'Your score: {int(player_score)}')
    print(f'Opponent score: {int(opp_score)}')

def print_env(env: OthelloVectEnv):
    cur_player = env.cur_players[0].item()
    if cur_player == 0:
        print('O to play')
    else:
        print('X to play')

    legal_actions = env.get_legal_actions()[0].nonzero().flatten().tolist()

    for i in range(8):
        for j in range(8):
            action = (i*8) + j

            if env.states[0, 0, i, j] == 1:
                print('X' if cur_player else 'O', end='  ')
            elif env.states[0, 1, i, j] == 1:
                print('O' if cur_player else 'X', end='  ')
            elif action in legal_actions:
                print(str(action).zfill(2), end=' ')
            else:
                print('.', end='  ')
        print('\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='OthelloDemo')
    parser.add_argument('--num_iters', default=500)      # option that takes a value
    parser.add_argument('--checkpoint', default='checkpoints/16-2.pt')
    args = parser.parse_args()

    env = OthelloVectEnv(1, torch.device('cpu'), debug=True)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model = TurboZeroResnet(checkpoint['model_arch_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    evaluator = OthelloMCTS(1, torch.device('cpu'), 8, MCTSHypers(
        num_iters = int(args.num_iters),
        dirichlet_alpha=0.01,
        dirichlet_epsilon=0,
    ), env=env, debug=False)
    play(model, env, evaluator)
