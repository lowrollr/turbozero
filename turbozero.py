
from typing import Union
import torch
import logging
import argparse
from core.algorithms.load import init_evaluator, init_trainable_evaluator
from core.resnet import ResNetConfig, TurboZeroResnet
from core.test.tester import  Tester
from core.test.tournament.tournament import Tournament, TournamentPlayer
from core.train.trainer import Trainer, load_checkpoint, init_history


import yaml

from envs.load import init_collector, init_env, init_tester, init_trainer


def load_trainer_nb(
    config_file: str,
    gpu: bool,
    debug: bool,
    logfile: str = '',
    verbose_logging: bool = True,
    checkpoint: str = ''
) -> Trainer: 
    args = argparse.Namespace(
        config=config_file,
        gpu=gpu,
        debug=debug,
        logfile=logfile,
        verbose=verbose_logging,
        checkpoint=checkpoint
    )
    
    if not args.logfile:
        args.logfile = 'turbozero.log'
    logging.basicConfig(filename=args.logfile, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')
    return load_trainer(args, interactive=True)

def load_tester_nb(
    config_file: str,
    gpu: bool,
    debug: bool,
    logfile: str = '',
    verbose_logging: bool = True,
    checkpoint: str = ''
) -> Tester:
    args = argparse.Namespace(
        config=config_file,
        gpu=gpu,
        debug=debug,
        logfile=logfile,
        verbose=verbose_logging,
        checkpoint=checkpoint
    )
    
    if not args.logfile:
        args.logfile = 'turbozero.log'
    logging.basicConfig(filename=args.logfile, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')
    return load_tester(args, interactive=True)

def load_tournament_nb(
    config_file: str,
    gpu: bool,
    debug: bool,
    logfile: str = '',
    verbose_logging: bool = True
) -> Tournament:
    args = argparse.Namespace(
        config=config_file,
        gpu=gpu,
        debug=debug,
        logfile=logfile,
        verbose=verbose_logging
    )
    if not args.logfile:
        args.logfile = 'turbozero.log'
    logging.basicConfig(filename=args.logfile, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')
    return load_tournament(args, interactive=True)
    

def load_config(config_file: str) -> dict:
    if config_file:
        with open(config_file, "r") as stream:
            raw_config = yaml.safe_load(stream)
    else:
        print('No config file provided, please provide a config file with --config')
        exit(1)
    return raw_config


def load_trainer(args, interactive: bool) -> Trainer:
    raw_config = load_config(args.config)

    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    episode_memory_device = torch.device('cpu') # we do not support episdoe memory on GPU yet

    if args.checkpoint:
        model, optimizer, history, run_tag, train_config, env_config = load_checkpoint(args.checkpoint)
        model = model.to(device)
        run_tag = raw_config.get('run_tag', run_tag)
    else:
        env_config = raw_config['env_config']
        train_config = raw_config['train_mode_config']
        run_tag = raw_config.get('run_tag', '')
        model = TurboZeroResnet(ResNetConfig(**raw_config['model_config'])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=raw_config['train_mode_config']['learning_rate'])
        history = init_history()

    env_type = env_config['env_type']
    parallel_envs_train = train_config['parallel_envs']
    parallel_envs_test = train_config['test_config']['episodes_per_epoch']
    env_train = init_env(device, parallel_envs_train, env_config, args.debug)
    env_test = init_env(device, parallel_envs_test, env_config, args.debug)
    train_evaluator = init_trainable_evaluator(train_config['algo_config'], env_train, model)
    train_collector = init_collector(episode_memory_device, env_type, train_evaluator)
    test_evaluator = init_trainable_evaluator(train_config['test_config']['algo_config'], env_test, model)
    test_collector = init_collector(episode_memory_device, env_type, test_evaluator)
    tester = init_tester(train_config['test_config'], test_collector, model, history, optimizer, args.verbose)
    trainer = init_trainer(device, env_type, train_collector, tester, model, optimizer, train_config, env_config, history, args.verbose, interactive, run_tag)
    return trainer

def load_tester(args, interactive: bool) -> Tester:
    raw_config = load_config(args.config)
    test_config = raw_config['test_mode_config']

    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    episode_memory_device = torch.device('cpu')

    if args.checkpoint:
        model, _, history, run_tag, _, env_config = load_checkpoint(args.checkpoint)
        
    else:
        print('No checkpoint provided, please provide a checkpoint with --checkpoint')
        exit(1)

    parallel_envs = test_config['episodes_per_epoch']
    model = model.to(device)
    run_tag = raw_config.get('run_tag', run_tag)
    env = init_env(device, parallel_envs, env_config, args.debug)
    evaluator = init_trainable_evaluator(test_config['algo_config'], env, model)
    collector = init_collector(episode_memory_device, env_config['env_type'], evaluator)
    tester = init_tester(test_config, collector, model, history, None, args.verbose)
    return tester

def load_tournament(args, interactive: bool) -> Tournament:
    raw_config = load_config(args.config)
    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    tournament_config = raw_config['tournament_mode_config']
    env = init_env(device, tournament_config['num_games'], raw_config['env_config'], args.debug)

    competitors = []
    for competitor in tournament_config['competitors']:
        if competitor.get('checkpoint'):
            model, _, _, _, _, _ = load_checkpoint(competitor['checkpoint'])
            model = model.to(device)
            evaluator = init_evaluator(competitor['algo_config'], env, model)
        else:
            evaluator = init_evaluator(competitor['algo_config'], env)
        player = TournamentPlayer(competitor['name'], evaluator)
        competitors.append(player)
    tournament = Tournament(competitors, env, tournament_config['num_games'], tournament_config['num_tournaments'])
    return tournament


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TurboZero')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--mode', type=str, default='demo', choices=['train', 'evaluate', 'demo'])
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--interactive', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--logfile', type=str, default='')
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as stream:
            raw_config = yaml.safe_load(stream)
    else:
        print('No config file provided, please provide a config file with --config')
        exit(1)

    if not args.logfile:
        args.logfile = 'turbozero.log'
    logging.basicConfig(filename=args.logfile, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')

    if args.mode == 'train':
        trainer = load_trainer(args, interactive=False)
        trainer.training_loop()
    elif args.mode == 'test':
        tester = load_tester(args, interactive=False)
        tester.collect_test_batch()
    elif args.mode == 'tournament':
        tournament = load_tournament(args)
        tournament.run()

