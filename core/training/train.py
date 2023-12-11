

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
    test_episodes: int
    checkpoint_every_n_epochs: int
    checkpoint_dir: str
    retain_n_checkpoints: int
    learning_rate: float
    momentum: float
    l2_lambda: float
    policy_factor: float
    disk_store_location: str # where to store env and evaluator states when not in use
    retain_n_checkpoints: int
    max_episode_steps: int

@struct.dataclass
class BaseExperience(struct.PyTreeNode):
    observation: struct.PyTreeNode
    policy: jnp.ndarray
    policy_mask: jnp.ndarray
    player_id: jnp.ndarray

@struct.dataclass
class BNTrainState(train_state.TrainState):
    batch_stats: Optional[Any] = None

@struct.dataclass
class TrainerState:
    key: jax.random.PRNGKey
    evaluator_state: EvaluatorState
    env_state: EnvState
    buff_state: EndRewardReplayBufferState
    epoch: int
    train_state: BNTrainState

@struct.dataclass
class TestState:
    trainer_state: TrainerState
    test_env_state: EnvState
    test_eval_state: EvaluatorState
    rewards: jnp.ndarray
    completed: jnp.ndarray

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
        
        self.batch_size = self.config.selfplay_batch_size
        self.test_batch_size = self.config.test_episodes

        self.pkl_file = 'state.pkl'


        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=self.config.retain_n_checkpoints, create=True)
        self.checkpoint_mngr = orbax.checkpoint.CheckpointManager(
            self.config.checkpoint_dir,
            checkpointer, 
            options=options
        )

        self.train_in_memory = True

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

        init_key, buff_key, model_key, env_key, eval_key = jax.random.split(key, 5)
        env_keys = jax.random.split(env_key, self.buff.config.batch_size)

        eval_keys = jax.random.split(eval_key, self.buff.config.batch_size)
        env_state, _ = jax.vmap(self.env.reset)(env_keys)
        eval_state = jax.vmap(self.train_evaluator.reset)(eval_keys)

        buff_state = self.buff.init(
            buff_key,
            jax.tree_map(
                lambda x: jnp.zeros(x.shape[1:], x.dtype),
                BaseExperience(
                    observation=env_state._observation,
                    policy=jax.vmap(self.train_evaluator.get_policy)(eval_state),
                    policy_mask=env_state.legal_action_mask,
                    player_id=env_state.cur_player_id
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
            key=init_key,
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
        player_id = env_state.cur_player_id
        evaluation_fn = partial(self.train_evaluator.evaluate, model_params=self.pack_model_params(state))
        eval_state = jax.vmap(evaluation_fn)(eval_state, env_state)
        eval_state, action = jax.vmap(self.train_evaluator.choose_action)(eval_state, env_state)
        env_state, terminated = jax.vmap(self.env.step)(env_state, action)

        buff_state = self.buff.add_experience(
            buff_state,
            BaseExperience(
                observation=observation,
                policy=jax.vmap(self.train_evaluator.get_policy)(eval_state),
                policy_mask=env_state.legal_action_mask,
                player_id=player_id
            )
        )

        buff_state = self.buff.assign_rewards(
            buff_state, 
            rewards=env_state.reward, 
            select_batch=terminated
        )

        eval_state = jax.vmap(self.train_evaluator.step_evaluator)(eval_state, action, terminated)
        env_state, terminated = jax.vmap(self.env.reset_if_terminated)(env_state, terminated)

        return state.replace(
            env_state=env_state,
            evaluator_state=eval_state,
            buff_state=buff_state
        )
    
    def test_step(self,
        state: TestState
    ) -> Tuple[TestState, Any]: 
        env_state, eval_state = state.trainer_state.env_state, state.trainer_state.evaluator_state
        
        evaluation_fn = partial(self.test_evaluator.evaluate, model_params=self.pack_model_params(state.trainer_state))
        eval_state = jax.vmap(evaluation_fn)(eval_state, env_state)
        eval_state, action = jax.vmap(self.test_evaluator.choose_action)(eval_state, env_state)
        env_state, terminated = jax.vmap(self.env.step)(env_state, action)
        eval_state = jax.vmap(self.test_evaluator.step_evaluator)(eval_state, action, terminated)

        return state.replace(
            trainer_state=state.trainer_state.replace(
                env_state=env_state,
                evaluator_state=eval_state
            ),
            reward = jnp.where(
                terminated & ~state.completed,
                env_state.reward,
                state.reward
            ),
            completed = state.completed | terminated
        ), None

    def train_step(self,
        state: TrainerState,       
    ) -> TrainerState:
        # sample from replay memory
        buff_state, batch, rewards = self.buff.sample(state.buff_state)
        batch = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32),
            batch
        )
        player_rewards = rewards[jnp.arange(self.buff.config.sample_size), batch.player_id.astype(jnp.int32)]

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
            value_loss = optax.l2_loss(pred_value.reshape(-1), player_rewards).mean()

            l2_reg = self.config.l2_lambda * jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_map(
                    lambda x: (x ** 2).sum(),
                    params
                )
            )

            loss = policy_loss + value_loss + l2_reg
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
            'value_accuracy': jnp.mean(jnp.round(pred_value) == jnp.round(player_rewards))
        }

        return state.replace(
            train_state = state.train_state.replace(
                batch_stats=updates['batch_stats'],
            ),
            buff_state=buff_state
        ), metrics
    
    def test(self,
        state: TrainerState,
    ) -> Tuple[TrainerState, Any]:
        # convert to test state
        env_key, eval_key, new_key = jax.random.split(state.key, 3)
        env_keys = jax.random.split(env_key, self.test_batch_size)
        eval_keys = jax.random.split(eval_key, self.test_batch_size)
        state = TestState(
            trainer_state=state.replace(key=new_key),
            test_env_state=jax.vmap(self.env.reset)(env_keys)[0],
            test_eval_state=jax.vmap(self.test_evaluator.reset)(eval_keys),
            rewards=jnp.zeros((self.test_batch_size,)),
            completed=jnp.zeros((self.test_batch_size,), dtype=jnp.bool_)
        )
        
        # make test steps
        state, _ = jax.lax.scan(
            lambda s, _: self.test_step(s),
            state,
            jnp.arange(self.config.max_episode_steps)
        )

        metrics = {
            'reward': state.reward.mean() 
        }

        return state.trainer_state, metrics
    
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
    
    def warmup(self,
        state: TrainerState,           
    ) -> TrainerState:
        return jax.lax.scan(
            lambda s, _: (jax.jit(self.collect_step)(s), None),
            state,
            jnp.arange(self.config.warmup_steps)
        )[0]
    
    def train_loop(
        self,
        state: TrainerState,
        num_epochs: int,
        warmup=True
    ) -> TrainerState:
        if warmup:
            state = self.warmup(state)

        train_epoch = jax.jit(self.train_epoch)
        test = jax.jit(self.test)
        for _ in range(num_epochs):
            # train
            state, metrics = train_epoch(state)
            if not self.debug:
                wandb.log({f'train/{k}': v.mean() for k, v in metrics.items()})
            # evaluate
            if state.epoch % self.config.test_every_n_epochs == 0:
                state, metrics = test(state)
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
            'eval_config_train': self.train_evaluator.config.__dict__,
            'eval_config_test': self.test_evaluator.config.__dict__,
            'buff_config': self.buff.config.__dict__,
            'model_config': self.model.config.__dict__
        }

        save_args = orbax_utils.save_args_from_target(ckpt)

        self.checkpoint_mngr.save(
            state.epoch,
            ckpt, save_kwargs={'save_args': save_args}
        )