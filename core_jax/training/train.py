

from functools import partial
from typing import Any, Dict, Optional, Tuple
from flax import struct
from flax.training import train_state
from dataclasses import dataclass
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
from core_jax.collector import BaseExperience, Collector, CollectorConfig, CollectorState
from core_jax.envs.env import Env, EnvState

from core_jax.evaluators.evaluator import Evaluator, EvaluatorState
from core_jax.evaluators.nn_evaluator import NNEvaluator
from core_jax.utils.replay_memory import EndRewardReplayBuffer, EndRewardReplayBufferState



@dataclass
class TrainerConfig(CollectorConfig):
    warmup_steps: int
    collection_steps_per_epoch: int
    train_steps_per_epoch: int
    epochs_per_checkpoint: int
    learning_rate: float
    momentum: float
    policy_factor: float

@struct.dataclass
class TrainerState(CollectorState, train_state.TrainState):
    batch_stats: Optional[Any] = None

class Trainer(Collector):
    def __init__(self,
        config: TrainerConfig,
        env: Env,
        evaluator: NNEvaluator,
        buff: EndRewardReplayBuffer,
        model: nn.Module
    ):
        super().__init__(config, env, evaluator, buff)
        self.model = model
    
    def pack_model_params(
        self,
        state: TrainerState
    )  -> Dict:
        params = {'params': state.params}
        if state.batch_stats is not None:
            params['batch_stats'] = state.batch_stats
        return params
    
    def init(self, 
        key: jax.random.PRNGKey,
        env_state: EnvState, 
        eval_state: EvaluatorState, 
        model_params: Optional[struct.PyTreeNode] = None
    ) -> TrainerState:
        buff_state = self.buff.init(
            jax.tree_map(
                lambda x: jnp.zeros(x.shape[1:], x.dtype),
                BaseExperience(
                    observation=env_state._observation,
                    policy=jax.vmap(self.evaluator.get_policy)(eval_state),
                    policy_mask=env_state.legal_action_mask
                )
            )
        )
        
        if model_params is None:
            model_params = self.model.init(key, jnp.zeros((1, *self.env.get_observation_shape())), train=False)

        return TrainerState.create(
            apply_fn=self.model.apply,
            params=model_params['params'],
            batch_stats=model_params.get('batch_stats'),
            evaluator_state=eval_state,
            env_state=env_state,
            buff_state=buff_state,
            tx=optax.sgd(learning_rate=self.config.learning_rate, momentum=self.config.momentum)
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
            (pred_policy, pred_value), updates = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats}, 
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
        (loss, ((policy_loss, value_loss, pred_policy, pred_value), updates)), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(
            batch_stats=updates['batch_stats'],
            buff_state=buff_state
        ) 
        metrics = {
            'loss': loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_accuracy': jnp.mean(jnp.argmax(pred_policy, axis=-1) == jnp.argmax(batch.policy, axis=-1)),
            'value_accuracy': jnp.mean(jnp.round(pred_value) == jnp.round(rewards))
        }

        return state, metrics

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
        



        
    



