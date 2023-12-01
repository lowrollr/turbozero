

from functools import partial
from typing import Any, Dict, Optional, Tuple
from flax import struct, serialization
from flax.training import train_state, checkpoints, orbax_utils
from dataclasses import dataclass
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from core.collector import BaseExperience, Collector, CollectorConfig, CollectorState
from core.envs.env import Env, EnvConfig, EnvState
from core.envs.make import make_env

from core.evaluators.evaluator import Evaluator, EvaluatorConfig, EvaluatorState
from core.evaluators.make import make_evaluator
from core.evaluators.nn_evaluator import NNEvaluator
from core.memory.make import make_replay_buffer
from core.networks.make import make_model
from core.memory.replay_memory import EndRewardReplayBuffer, EndRewardReplayBufferState
import wandb




@dataclass
class TrainerConfig(CollectorConfig):
    warmup_steps: int
    collection_steps_per_epoch: int
    train_steps_per_epoch: int
    epochs_per_checkpoint: int
    checkpoint_dir: str
    max_checkpoints_to_keep: int
    learning_rate: float
    momentum: float
    policy_factor: float


@struct.dataclass
class BNTrainState(train_state.TrainState):
    batch_stats: Optional[Any] = None

@struct.dataclass
class TrainerState(CollectorState):
    epoch: int
    train_state: BNTrainState

class Trainer(Collector):
    def __init__(self,
        config: TrainerConfig,
        env: Env,
        evaluator: NNEvaluator,
        buff: EndRewardReplayBuffer,
        model: nn.Module,

    ):
        super().__init__(config, env, evaluator, buff)
        self.config: TrainerConfig
        self.model = model

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=self.config.max_checkpoints_to_keep)
        self.checkpoint_mngr = orbax.checkpoint.CheckpointManager(
            self.config.checkpoint_dir,
            checkpointer, 
            options=options
        )

    
    def pack_model_params(
        self,
        state: TrainerState
    )  -> Dict:
        params = {'params': state.train_state.params}
        if state.train_state.batch_stats is not None:
            params['batch_stats'] = state.train_state.batch_stats
        return params
    
    def init(self, 
        key: jax.random.PRNGKey,
    ) -> TrainerState:
        buff_key, model_key, env_key, eval_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.buff.config.batch_size)
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
    ) -> CollectorState:
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
        if warmup:
            state = self.warmup(state)

        for _ in range(num_epochs):
            # train
            state, metrics = self.train_epoch(state)
            wandb.log({k: v.mean() for k, v in metrics.items()})
            # evaluate
            # TODO

            # checkpoint
            if state.epoch % self.config.epochs_per_checkpoint == 0:
                self.save_checkpoint(state)

            state = state.replace(epoch=state.epoch + 1)

        return state
    
    def save_checkpoint(
        self,
        state: TrainerState,
    ) -> None:
        
        configs = {
            'train_config': self.config,
            'env_config': self.env.config,
            'eval_config': self.evaluator.config,
            'buff_config': self.buff.config,
            'model_config': self.model.config
        }

        ckpt = {
            'state': state.train_state,
            'config': configs
        }

        save_args = orbax_utils.save_args_from_target(ckpt)

        self.checkpoint_mngr.save(
            state.epoch,
            ckpt, save_kwargs={'save_args': save_args}
        )


def init_from_checkpoint(
    checkpoint_dir: str,
    checkpoint_num: Optional[int],
    seed: int = 0
) -> Tuple[Trainer, TrainerState]:
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_mngr = orbax.checkpoint.CheckpointManager(
        checkpoint_dir,
        checkpointer=checkpointer
    )
    if checkpoint_num is None:
        checkpoint_num = checkpoint_mngr.latest_step()
    
    ckpt = checkpoint_mngr.restore(checkpoint_num)
    
    train_state = ckpt['state']
    configs = ckpt['config']

    env = make_env(configs['env_config'])
    model = make_model(configs['model_config'])
    evaluator = make_evaluator(configs['eval_config'], env, model)
    buff = make_replay_buffer(configs['buff_config'])

    trainer = Trainer(
        configs['train_config'],
        env,
        evaluator,
        buff,
        model
    )

    trainer_state = trainer.init(jax.random.PRNGKey(seed))
    
    trainer_state = trainer_state.replace(
        train_state=train_state
    )

    return trainer, trainer_state

        










    
        



        
    



