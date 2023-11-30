

from functools import partial
from typing import Any, Dict, Optional
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
    
    def unpack_model_params(
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
            BaseExperience(
                observation=env_state._observation,
                policy=jax.vmap(self.evaluator.get_policy)(eval_state),
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
        evaluation_fn = partial(self.evaluator.evaluate, model_params=self.unpack_model_params(state))
        eval_state = jax.vmap(evaluation_fn)(eval_state, env_state)
        eval_state, action = jax.vmap(self.evaluator.choose_action)(eval_state, env_state)
        env_state, terminated = jax.vmap(self.env.step)(env_state, action)

        buff_state = self.buff.add_experience(
            buff_state,
            BaseExperience(
                observation=observation,
                policy=jax.vmap(self.evaluator.get_policy)(eval_state),
            )
        )

        eval_state = jax.vmap(self.evaluator.step_evaluator)(eval_state, action, terminated)
        env_state, terminated = jax.vmap(self.env.reset_if_terminated)(env_state, terminated)

        return state.replace(
            env_state=env_state,
            evaluator_state=eval_state,
            buff_state=buff_state
        )
    
   

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
    ) -> TrainerState:
        state, _ = jax.lax.scan(
            lambda s, _: (self.collect_step(s), None),
            state,
            jnp.arange(self.config.collection_steps_per_epoch)
        )

        return state



        
    



