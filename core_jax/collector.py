
from functools import partial
import jax
import jax.numpy as jnp

from flax import struct
from dataclasses import dataclass

from core_jax.envs.env import EnvState, Env
from core_jax.evaluators.evaluator import Evaluator, EvaluatorState
from core_jax.utils.replay_memory import EndRewardReplayBuffer, EndRewardReplayBufferState


@dataclass
class CollectorConfig:
    pass

@struct.dataclass
class CollectorState:
    evaluator_state: EvaluatorState
    env_state: EnvState
    buff_state: EndRewardReplayBufferState

@struct.dataclass
class BaseExperience(struct.PyTreeNode):
    observation: struct.PyTreeNode
    policy: jnp.ndarray


class Collector:
    def __init__(self, 
        config: CollectorConfig,
        env: Env,
        evaluator: Evaluator,
        buff: EndRewardReplayBuffer 
    ): 
        self.config = config
        self.env = env
        self.evaluator = evaluator
        self.buff = buff

        self.evaluation_fn = self.evaluator.evaluate

    def init(self, env_state, eval_state) -> CollectorState:
        buff_state = self.buff.init(
            BaseExperience(
                observation=env_state._observation,
                policy=jax.vmap(self.evaluator.get_policy)(eval_state),
            )
        )

        return CollectorState(
            evaluator_state=eval_state,
            env_state=env_state,
            buff_state=buff_state
        )

    def collect_step(
        self,
        state: CollectorState,
        eval_args: dict,
    ) -> CollectorState:
        env_state, eval_state, buff_state = state.env_state, state.evaluator_state, state.buff_state
        observation = env_state._observation
        evaluation_fn = partial(self.evaluation_fn, **eval_args)
        eval_state = jax.vmap(evaluation_fn, in_axes=(0,0))(eval_state, env_state)
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
