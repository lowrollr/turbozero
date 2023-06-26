



from typing import Optional

import torch
from core import GLOB_FLOAT_TYPE
from core.history import Metric, TrainingMetrics
from core.hyperparameters import LZHyperparameters
from core.memory import GameReplayMemory
from core.vecttrainer import VectTrainer
from core.lz_resnet import LZArchitectureParameters, LZResnet
from envs.othello.vectenv import OthelloVectEnv
from envs.othello.vectmcts import OthelloLazyMCTS

class OthelloTrainer(VectTrainer):
    def __init__(self,
        train_evaluator: OthelloLazyMCTS,
        test_evaluator: Optional[OthelloLazyMCTS],
        model: LZResnet,
        optimizer: torch.optim.Optimizer,
        hypers: LZHyperparameters,
        num_parallel_envs: int,
        device: torch.device,
        history: Optional[TrainingMetrics] = None,
        memory: Optional[GameReplayMemory] = None,
        log_results: bool = True,
        interactive: bool = True,
        run_tag: Optional[str] = None
    ):
        run_tag = run_tag or 'othello'
        memory = memory or GameReplayMemory(hypers.replay_memory_size)
        super().__init__(
            train_evaluator=train_evaluator,
            test_evaluator=test_evaluator,
            model=model,
            optimizer=optimizer,
            hypers=hypers,
            num_parallel_envs=num_parallel_envs,
            device=device,
            history=history,
            memory=memory,
            log_results=log_results,
            interactive=interactive,
            run_tag=run_tag
        )

        self.test_evaluator: OthelloLazyMCTS
        self.train_evaluator: OthelloLazyMCTS

    def init_history(self):
        return TrainingMetrics(
            train_metrics=[
                Metric(name='loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results),
                Metric(name='value_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Value Loss'),
                Metric(name='policy_loss', xlabel='Step', ylabel='Loss', addons={'running_mean': 100}, maximize=False, alert_on_best=self.log_results, proper_name='Policy Loss'),
                Metric(name='policy_accuracy', xlabel='Step', ylabel='Accuracy (%)', addons={'running_mean': 100}, maximize=True, alert_on_best=self.log_results, proper_name='Policy Accuracy'),
            ],
            episode_metrics=[],
            eval_metrics=[],
            epoch_metrics=[]
        )
    
    def push_examples_to_memory_buffer(self, terminated_envs, is_eval, info):
        for t in terminated_envs:
            reward = info['rewards'][t].item()
            if is_eval:
                self.unfinished_episodes_test[t] = []
                
            else:
                game = []
                for (s, v) in self.unfinished_episodes_train[t]:
                    game.append((s, v, reward))
                self.memory.insert(game)
                self.unfinished_episodes_train[t] = []

    def add_collection_metrics(self, terminated_envs, is_eval):
        pass

    def add_train_metrics(self, policy_loss, value_loss, policy_accuracy, loss):
        self.history.add_training_data({
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_accuracy': policy_accuracy,
            'loss': loss
        }, log=self.log_results)

    def add_epoch_metrics(self):
        pass

def init_new_othello_trainer(
        arch_params: LZArchitectureParameters,
        parallel_envs: int,
        device: torch.device,
        hypers: LZHyperparameters,
        log_results=True,
        interactive=True,
        run_tag: Optional[str] = None
    ) -> OthelloTrainer:
        model = LZResnet(arch_params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hypers.learning_rate)

        train_evaluator = OthelloLazyMCTS(OthelloVectEnv(parallel_envs, device), hypers.mcts_c_puct)
        test_evaluator = OthelloLazyMCTS(OthelloVectEnv(parallel_envs, device), hypers.mcts_c_puct)

        return OthelloTrainer(train_evaluator, test_evaluator, model, optimizer, hypers, parallel_envs, device, None, None, log_results, interactive, run_tag)

def init_othello_trainer_from_checkpoint(
        parallel_envs: int, 
        checkpoint_path: str, 
        device: torch.device, 
        memory: Optional[GameReplayMemory] = None, 
        log_results=True, 
        interactive=True
    ) -> OthelloTrainer:
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hypers: LZHyperparameters = checkpoint['hypers']
    model = LZResnet(checkpoint['model_arch_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hypers.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    history = checkpoint['history']
    history.reset_all_figs()
    run_tag = checkpoint['run_tag']

    train_evaluator = OthelloLazyMCTS(OthelloVectEnv(parallel_envs, device), hypers.mcts_c_puct)
    test_evaluator = OthelloLazyMCTS(OthelloVectEnv(parallel_envs, device), hypers.mcts_c_puct)
    
    trainer = OthelloTrainer(train_evaluator, test_evaluator, model, optimizer, hypers, parallel_envs, device, history, memory, log_results, interactive, run_tag=run_tag)
    return trainer