

from functools import partial
import os
from typing import Any, Dict, Optional, Tuple
from flax import struct, serialization
from flax.training import train_state, checkpoints, orbax_utils
from dataclasses import dataclass
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from core.envs.env import Env, EnvConfig, EnvState
import pickle
from core.envs.make import make_env

from core.evaluators.evaluator import Evaluator, EvaluatorConfig, EvaluatorState
from core.evaluators.make import make_evaluator
from core.evaluators.nn_evaluator import NNEvaluator
from core.memory.make import make_replay_buffer
from core.networks.make import make_model
from core.memory.replay_memory import EndRewardReplayBuffer, EndRewardReplayBufferState
import wandb

import yaml

@dataclass
class TrainerConfig:
    selfplay_batch_size: int
    warmup_steps: int
    collection_steps_per_epoch: int
    train_steps_per_epoch: int
    test_every_n_epochs: int
    test_batch_size: int
    test_episodes: int
    checkpoint_every_n_epochs: int
    checkpoint_dir: str
    retain_n_checkpoints: int
    learning_rate: float
    momentum: float
    policy_factor: float
    disk_store_location: str # where to store env and evaluator states when not in use
    retain_n_checkpoints: int

@struct.dataclass
class BaseExperience(struct.PyTreeNode):
    observation: struct.PyTreeNode
    policy: jnp.ndarray
    policy_mask: jnp.ndarray


@struct.dataclass
class BNTrainState(train_state.TrainState):
    batch_stats: Optional[Any] = None

@struct.dataclass
class TrainerState:
    evaluator_state: EvaluatorState
    env_state: EnvState
    buff_state: EndRewardReplayBufferState
    epoch: int
    train_state: BNTrainState

class Trainer:
    def __init__(self,
        config: TrainerConfig,
        env: Env,
        train_evaluator: NNEvaluator,
        test_evaluator: NNEvaluator,
        buff: EndRewardReplayBuffer,
        model: nn.Module,
        debug: bool = False
    ):
        self.train_evaluator = train_evaluator
        self.test_evaluator = test_evaluator
        self.evaluator = train_evaluator
        self.env = env
        self.config = config
        self.config: TrainerConfig
        self.model = model
        self.buff = buff
        self.debug = debug

        self.collection_batch_size = buff.config.batch_size

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=self.config.retain_n_checkpoints, create=True)
        self.checkpoint_mngr = orbax.checkpoint.CheckpointManager(
            self.config.checkpoint_dir,
            checkpointer, 
            options=options
        )

        self.train_in_memory = True

    def get_swap_file_names(self, get_train: bool) -> Tuple[str, str]:
        write_file, read_file = (('test_state.pkl', 'train_state.pkl') if get_train else ('train_state.pkl', 'test_state.pkl'))
        
        write_file, read_file = self.config.disk_store_location + '/' + write_file, self.config.disk_store_location + '/' + read_file

        return write_file, read_file

    def swap_states(self,
        state: TrainerState,
        get_train: bool
    ) -> TrainerState:
        if not self.train_in_memory ^ get_train:
            return state
        
        write_file, read_file = self.get_swap_file_names(get_train)
        
        with open(write_file, 'wb') as f:
            pickle.dump(state.evaluator_state, f)

        with open(read_file, 'rb') as f:
            evaluator_state = pickle.load(f)

        self.train_in_memory = get_train
        return state.replace(
            evaluator_state=evaluator_state
        )
    
    def pack_model_params(
        self,
        state: TrainerState
    )  -> Dict:
        params = {'params': state.train_state.params}
        if state.train_state.batch_stats is not None:
            params['batch_stats'] = state.train_state.batch_stats
        return params
    

    def init(self, # no vmapping this!
        key: jax.random.PRNGKey,
    ) -> TrainerState:
        self.train_in_memory = True
        self.evaluator = self.train_evaluator

        buff_key, model_key, env_key, eval_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.buff.config.batch_size)

        test_eval_key, eval_key = jax.random.split(eval_key, 2)
        test_eval_keys = jax.random.split(test_eval_key, min(self.config.test_episodes, self.collection_batch_size))
        
        eval_state_test = jax.vmap(self.test_evaluator.reset)(test_eval_keys)
        train, _ = self.get_swap_file_names(True)

        with open(train, 'wb') as f:
            pickle.dump(eval_state_test, f)

        eval_keys = jax.random.split(eval_key, self.buff.config.batch_size)
        env_state, _ = jax.vmap(self.env.reset)(env_keys)
        eval_state = jax.vmap(self.evaluator.reset)(eval_keys)

        buff_state = self.buff.init(
            buff_key,
            jax.tree_map(
                lambda x: jnp.zeros(x.shape[1:], x.dtype),
                BaseExperience(
                    observation=env_state._observation,
                    policy=jax.vmap(self.evaluator.get_policy)(eval_state),
                    policy_mask=env_state.legal_action_mask
                )
            )
        )

        model_params = self.model.init(model_key, jnp.zeros((1, *self.env.get_observation_shape())), train=False)

        train_state = BNTrainState.create(
            apply_fn=self.model.apply,
            params=model_params['params'],
            batch_stats=model_params.get('batch_stats'),
            tx=optax.sgd(learning_rate=self.config.learning_rate, momentum=self.config.momentum)
        )

        return TrainerState(
            epoch=0,
            evaluator_state=eval_state,
            env_state=env_state,
            buff_state=buff_state,
            train_state=train_state
        )

    def collect_step(self,
        state: TrainerState,
    ) -> TrainerState:
        env_state, eval_state, buff_state = state.env_state, state.evaluator_state, state.buff_state
        observation = env_state._observation
        evaluation_fn = partial(self.evaluator.evaluate, model_params=self.pack_model_params(state))
        eval_state = jax.vmap(evaluation_fn)(eval_state, env_state)
        eval_state, action = jax.vmap(self.evaluator.choose_action)(eval_state, env_state)
        env_state, terminated = jax.vmap(self.env.step)(env_state, action)

        buff_state = self.buff.add_experience(
            buff_state,
            BaseExperience(
                observation=observation,
                policy=jax.vmap(self.evaluator.get_policy)(eval_state),
                policy_mask=env_state.legal_action_mask
            )
        )

        buff_state = self.buff.assign_rewards(
            buff_state, 
            rewards=env_state.reward, 
            select_batch=terminated
        )

        eval_state = jax.vmap(self.evaluator.step_evaluator)(eval_state, action, terminated)
        env_state, terminated = jax.vmap(self.env.reset_if_terminated)(env_state, terminated)

        return state.replace(
            env_state=env_state,
            evaluator_state=eval_state,
            buff_state=buff_state
        )
    
    def train_step(self,
        state: TrainerState,       
    ) -> TrainerState:
        # sample from replay memory
        buff_state, batch, rewards = self.buff.sample(state.buff_state)
        batch = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32),
            batch
        )
        rewards = rewards.astype(jnp.float32)

        def loss_fn(params: struct.PyTreeNode):
            (pred_policy, pred_value), updates = state.train_state.apply_fn(
                {'params': params, 'batch_stats': state.train_state.batch_stats}, 
                x=batch.observation,
                train=True,
                mutable=['batch_stats']
            )
            pred_policy = jnp.where(
                batch.policy_mask,
                pred_policy,
                0
            )
            policy_loss = optax.softmax_cross_entropy(pred_policy, batch.policy).mean() * self.config.policy_factor
            value_loss = optax.l2_loss(pred_value, rewards).mean()

            loss = policy_loss + value_loss
            return loss, ((policy_loss, value_loss, pred_policy, pred_value), updates)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, ((policy_loss, value_loss, pred_policy, pred_value), updates)), grads = grad_fn(state.train_state.params)
        state = state.replace(
            train_state = state.train_state.apply_gradients(grads=grads)
        )
        metrics = {
            'loss': loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_accuracy': jnp.mean(jnp.argmax(pred_policy, axis=-1) == jnp.argmax(batch.policy, axis=-1)),
            'value_accuracy': jnp.mean(jnp.round(pred_value) == jnp.round(rewards))
        }

        return state.replace(
            train_state = state.train_state.replace(
                batch_stats=updates['batch_stats'],
            ),
            buff_state=buff_state
        ), metrics

    def warmup(self,
        state: TrainerState,           
    ) -> TrainerState:
        return jax.lax.scan(
            lambda s, _: (self.collect_step(s), None),
            state,
            jnp.arange(self.config.warmup_steps)
        )[0]
    
    def train_epoch(self,
        state: TrainerState,
    ) -> Tuple[TrainerState, Any]:
        # make collection steps
        state, _ = jax.lax.scan(
            lambda s, _: (self.collect_step(s), None),
            state,
            jnp.arange(self.config.collection_steps_per_epoch)
        )
        # then make train steps
        return jax.lax.scan(
            lambda s, _: self.train_step(s),
            state,
            jnp.arange(self.config.train_steps_per_epoch)
        )
    
    def train_loop(
        self,
        state: TrainerState,
        num_epochs: int,
        warmup=True
    ) -> TrainerState:
        state = self.swap_states(state, get_train=True)
        if warmup:
            state = self.warmup(state)

        for _ in range(num_epochs):
            # train
            state = self.swap_states(state, get_train=True)
            state, metrics = self.train_epoch(state)
            if not self.debug:
                wandb.log({f'train/{k}': v.mean() for k, v in metrics.items()})
            # evaluate
            if state.epoch % self.config.test_every_n_epochs == 0:
                # load test evaluator
                state = self.swap_states(state, get_train=False)
                # state, metrics = self.evaluate(state)
                if not self.debug:
                    wandb.log({f'test/{k}': v.mean() for k, v in metrics.items()})


            # checkpoint
            if state.epoch % self.config.checkpoint_every_n_epochs == 0:
                self.save_checkpoint(state)

            state = state.replace(epoch=state.epoch + 1)

        return state
    
    def save_checkpoint(
        self,
        state: TrainerState,
    ) -> None:
        ckpt = {
            'state': state.train_state,
            'train_config': self.config.__dict__,
            'env_config': self.env.config.__dict__,
            'eval_config': self.evaluator.config.__dict__,
            'buff_config': self.buff.config.__dict__,
            'model_config': self.model.config.__dict__
        }

        save_args = orbax_utils.save_args_from_target(ckpt)

        self.checkpoint_mngr.save(
            state.epoch,
            ckpt, save_kwargs={'save_args': save_args}
        )




def init_from_config(
    config: dict,
    debug=False
) -> Tuple[Trainer, TrainerState]:
    env_config = config['env_config']
    env = make_env(
        env_pkg=env_config['pkg'],
        env_name=env_config['name'],
        config=env_config['config']
    )

    evaluator_config = config['evaluator_config']
    evaluator_type = evaluator_config['type']
    evaluator_config_train = evaluator_config['train']
    evaluator_config_test = evaluator_config['test']
    evaluator_model_config = evaluator_config['model']
    model_type = evaluator_model_config['type']
    model_config = evaluator_model_config['config']

    model_config.update(
        policy_head_out_size=env.num_actions,
        value_head_out_size=1
    )

    model = make_model(
        model_type=model_type,
        config=model_config
    )

    evaluator_train = make_evaluator(
        evaluator_type=evaluator_type,
        config=evaluator_config_train, 
        env=env,
        model=model
    )

    evaluator_test = make_evaluator(
        evaluator_type=evaluator_type,
        config=evaluator_config_test,
        env=env,
        model=model
    )

    train_config = config['training_config']
    batch_size = train_config['selfplay_batch_size']

    buff_config = train_config['replay_buffer_config']
    buff = make_replay_buffer(
        buff_type=buff_config['type'],
        batch_size=batch_size,
        config=buff_config['config']
    )

    trainer_config = TrainerConfig(
        warmup_steps=train_config['warmup_steps'],
        collection_steps_per_epoch=train_config['collection_steps_per_epoch'],
        train_steps_per_epoch=train_config['train_steps_per_epoch'],
        test_every_n_epochs=train_config['test_every_n_epochs'],
        test_episodes=train_config['test_episodes'],
        checkpoint_every_n_epochs=train_config['checkpoint_every_n_epochs'],
        checkpoint_dir=train_config['checkpoint_dir'],
        retain_n_checkpoints=train_config['retain_n_checkpoints'],
        learning_rate=train_config['learning_rate'],
        momentum=train_config['momentum'],
        policy_factor=train_config['policy_factor'],
        disk_store_location=train_config['disk_store_location'],
        selfplay_batch_size=batch_size,
        test_batch_size=train_config['test_batch_size']
    )

    trainer = Trainer(
        trainer_config,
        env,
        evaluator_train,
        evaluator_test,
        buff,
        model,
        debug=debug
    )

    return trainer, trainer.init(jax.random.PRNGKey(train_config['seed']))


def init_from_config_file(
    config_file: str,
    debug=False
) -> Tuple[Trainer, TrainerState]:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return init_from_config(config, debug)


def init_from_checkpoint(
    checkpoint_dir: str,
    checkpoint_num: Optional[int] = None,
    debug=False
) -> Tuple[Trainer, TrainerState]:
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_mngr = orbax.checkpoint.CheckpointManager(
        checkpoint_dir,
        checkpointer
    )
    if checkpoint_num is None:
        checkpoint_num = checkpoint_mngr.latest_step()

    # load once to get configs
    ckpt = checkpoint_mngr.restore(checkpoint_num)

    env = make_env(ckpt['env_config'])
    model = make_model(ckpt['model_config'])
    evaluator = make_evaluator(ckpt['eval_config'], env, model=model)
    buff = make_replay_buffer(ckpt['buff_config'])

    trainer = Trainer(
        TrainerConfig(**ckpt['train_config']),
        env,
        evaluator,
        buff,
        model,
        debug=debug
    )

    # build target
    target = {
        'state': BNTrainState.create(
                apply_fn=model.apply,
                params=ckpt['state']['params'],
                batch_stats=ckpt['state'].get('batch_stats'),
                tx=optax.sgd(learning_rate=trainer.config.learning_rate, momentum=trainer.config.momentum)
        ),
        'train_config': None,
        'env_config': None,
        'eval_config': None,
        'buff_config': None,
        'model_config': None
    }

    # load again now that we have proper TrainState target (send help)
    ckpt = checkpoint_mngr.restore(checkpoint_num, items=target)

    trainer_state = trainer.init(jax.random.PRNGKey(trainer.config.seed))
    
    trainer_state = trainer_state.replace(
        train_state=ckpt['state']
    )

    return trainer, trainer_state
