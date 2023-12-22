

from functools import partial
import pickle
from typing import Any, Dict, Optional, Tuple
from flax import struct
import jax
import jax.numpy as jnp
from core.envs.env import Env, EnvState
from core.evaluators.evaluator import EvaluatorState
from core.evaluators.nn_evaluator import NNEvaluator
from core.memory.replay_memory import EndRewardReplayBuffer
from core.training.train import TestState, Trainer, TrainerConfig, TrainerState

import flax.linen as nn


@struct.dataclass
class TwoPlayerTrainerState(TrainerState):
    best_model_params: struct.PyTreeNode
    best_model_batch_stats: Optional[struct.PyTreeNode]

@struct.dataclass
class TwoPlayerTestState:
    trainer_state: TwoPlayerTrainerState
    test_env_state: EnvState
    test_eval_state: EvaluatorState
    opponent_eval_state: EvaluatorState
    outcomes: jnp.ndarray
    completed: jnp.ndarray
    
    
class TwoPlayerTrainer(Trainer):
    def __init__(self,
        config: TrainerConfig,
        env: Env,
        train_evaluator: NNEvaluator,
        test_evaluator: NNEvaluator,
        buff: EndRewardReplayBuffer,
        model: nn.Module,
        debug: bool = False
    ):
        super().__init__(config, env, train_evaluator, test_evaluator, buff, model, debug)        
        
    def convert_state(self, state: TrainerState) -> TwoPlayerTrainerState:
        return TwoPlayerTrainerState(
            **state.__dict__,
            best_model_params=state.train_state.params,
            best_model_batch_stats=state.train_state.batch_stats
        )
    
    def init(self, key: jax.random.PRNGKey) -> TwoPlayerTrainerState:
        return self.convert_state(super().init(key))
    
    def pack_best_model_params(self, state: TwoPlayerTrainerState) -> Dict:
        params = {'params': state.best_model_params}
        if state.best_model_batch_stats is not None:
            params['batch_stats'] = state.best_model_batch_stats
        return params
    
    def test_step(self, state: TestState, use_best_model: bool) -> Tuple[TestState, Any]:
        
        env_state = state.test_env_state
        model_params = jax.lax.cond(
            use_best_model,
            lambda: self.pack_best_model_params(state.trainer_state),
            lambda: self.pack_model_params(state.trainer_state)
        )

        evaluation_fn = partial(self.test_evaluator.evaluate, model_params=model_params)

        def evaluate_(ev, ov, en):
            ev = jax.vmap(evaluation_fn)(ev, en)
            ev, action = jax.vmap(self.test_evaluator.choose_action)(ev, en)
            en = jax.vmap(self.env.step)(en, action)
            ev = jax.vmap(self.test_evaluator.step_evaluator)(ev, action, en.terminated)
            ov = jax.vmap(self.test_evaluator.step_evaluator)(ov, action, en.terminated)
            return ev, ov, en, en.terminated

        active_eval_state, other_eval_state, env_state, terminated = jax.lax.cond(
            use_best_model,
            lambda: evaluate_(state.test_eval_state, state.opponent_eval_state, env_state),
            lambda: evaluate_(state.opponent_eval_state, state.test_eval_state, env_state)
        )
        
        return state.replace(
            test_env_state = env_state,
            test_eval_state = jax.lax.cond(
                use_best_model,
                lambda: other_eval_state,
                lambda: active_eval_state
            ),
            outcomes = jnp.where(
                (terminated & ~state.completed)[..., None],
                env_state.reward,
                state.outcomes
            ),
            completed = state.completed | terminated,
            opponent_eval_state = jax.lax.cond(
                use_best_model,
                lambda: active_eval_state,
                lambda: other_eval_state
            )
        ), None


    def test(self, 
        state: TwoPlayerTrainerState
    ) -> Tuple[TwoPlayerTrainerState, Any]:
        env_key, eval_key, new_key, op_eval_key = jax.random.split(state.key, 4)
        env_keys = jax.random.split(env_key, self.test_batch_size)
        eval_keys = jax.random.split(eval_key, self.test_batch_size)
        op_eval_keys = jax.random.split(op_eval_key, self.test_batch_size)
        
        test_env_state = jax.vmap(self.env.reset)(env_keys)[0]

        state = TwoPlayerTestState(
            trainer_state = state.replace(key=new_key),
            test_env_state = test_env_state,
            test_eval_state = jax.vmap(self.test_evaluator.reset)(eval_keys),
            opponent_eval_state = jax.vmap(self.test_evaluator.reset)(op_eval_keys),
            outcomes = jnp.zeros_like(test_env_state.reward),
            completed = jnp.zeros((self.test_batch_size,), dtype=jnp.bool_),
        )

        # make one test step
        state = self.test_step(state, True)[0]
        reset = jnp.zeros((self.test_batch_size,), dtype=jnp.bool_)
        reset = reset.at[:self.test_batch_size // 2].set(True)
        state = state.replace(
            test_env_state = jax.vmap(self.env.reset_if_terminated)(
                state.test_env_state, 
                reset
            )[0],
            test_eval_state = jax.vmap(self.test_evaluator.reset_if_terminated)(
                state.test_eval_state, 
                reset
            ),
            opponent_eval_state = jax.vmap(self.test_evaluator.reset_if_terminated)(
                state.opponent_eval_state,
                reset
            ),
            outcomes = jnp.where(
                reset[..., None],
                jnp.zeros_like(state.outcomes),
                state.outcomes
            ),
            completed = jnp.where(
                reset,
                jnp.zeros((self.test_batch_size,), dtype=jnp.bool_),
                state.completed
            )
        )
        
        new_model_player_id = state.test_env_state.cur_player_id

        state, _ = jax.lax.scan(
            self.test_step,
            state,
            jnp.tile(jnp.array([False, True]), self.config.max_episode_steps // 2)
        )

        cur_player_rewards = state.outcomes[jnp.arange(self.test_batch_size), new_model_player_id]

        win_margin = jnp.sum(cur_player_rewards)
        
        metrics = {
            'win_margin': win_margin / self.test_batch_size
        }

        return state.trainer_state.replace(
            best_model_params = jax.lax.cond(
                win_margin > 0,
                lambda: state.trainer_state.train_state.params,
                lambda: state.trainer_state.best_model_params
            ),
            best_model_batch_stats = jax.lax.cond(
                win_margin > 0,
                lambda: state.trainer_state.train_state.batch_stats,
                lambda: state.trainer_state.best_model_batch_stats
            )
        ), metrics
        
