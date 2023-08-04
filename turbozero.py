
import torch
import logging
import argparse
from core.algorithms.load import init_trainable_evaluator
from core.resnet import ResNetConfig, TurboZeroResnet
from core.test.tester import  Tester
from core.train.trainer import Trainer, init_history


import yaml

from envs.load import init_collector, init_env, init_tester, init_trainer

def run(args, interactive: bool):
    if not args.logfile:
        args.logfile = 'turbozero.log'

    logging.basicConfig(filename=args.logfile, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s')

    mode = args.mode
    if args.config:
        with open(args.config, "r") as stream:
            raw_config = yaml.safe_load(stream)
    else:
        print('No config file provided, please provide a config file with --config')
        exit(1)

    if torch.cuda.is_available() and args.gpu:
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    episode_memory_device = torch.device('cpu') # we do not support episdoe memory on GPU yet

    # load model checkpoint
    if mode == 'train':
        env_type = raw_config['env_type']
        # hack
        raw_config['env_config']['parallel_envs'] = raw_config['train_mode_config']['parallel_envs']
        env = init_env(device, env_type, raw_config['env_config'], args.debug)
        model = TurboZeroResnet(ResNetConfig(**raw_config['model_config']))
        optimizer = torch.optim.AdamW(model.parameters(), lr=raw_config['train_mode_config']['learning_rate'])
        evaluator = init_trainable_evaluator(raw_config['train_mode_config']['algo_type'], raw_config['train_mode_config']['algo_config'], env)
        collector = init_collector(episode_memory_device, env_type, evaluator)
        history = init_history()
        tester = init_tester(raw_config['train_mode_config']['test_config'], collector, model, optimizer, history, args.verbose)
        trainer = init_trainer(device, env_type, collector, tester, model, optimizer, raw_config['train_mode_config'], raw_config['env_config'], history, args.verbose, interactive, raw_config.get('run_tag', ''))
        trainer.training_loop()

    elif mode == 'evaluate':
        pass

    # elif mode == 'demo':

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
    run(args, interactive=False)

