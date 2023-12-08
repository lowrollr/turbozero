

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
    use_current_model: jnp.ndarray
    
    
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
    
    def test_step(self, state: TestState) -> Tuple[TestState, Any]:
        
        env_state = state.test_env_state

        best_model_params = self.pack_best_model_params(state.trainer_state)
        model_params = self.pack_model_params(state.trainer_state)

        def evaluate_(use_best, new_ev, old_ev, en):
            ev, ov = jax.lax.cond(
                use_best,
                lambda: (new_ev, old_ev),
                lambda: (old_ev, new_ev)
            )
            ev = jax.lax.cond(
                use_best,
                lambda : self.test_evaluator.evaluate(ev, en, model_params=best_model_params),
                lambda : self.test_evaluator.evaluate(ev, en, model_params=model_params)
            )
            ev, action = self.test_evaluator.choose_action(ev, en)
            en, terminated = self.env.step(en, action)
            ev = self.test_evaluator.step_evaluator(ev, action, terminated)
            ov = self.test_evaluator.step_evaluator(ov, action, terminated)

            new_ev, old_ev = jax.lax.cond(
                use_best,
                lambda: (ev, ov),
                lambda: (ov, ev)
            )

            return new_ev, old_ev, en, terminated
        
        new_model_ev_state, best_model_ev_state, env_state, terminated = jax.vmap(evaluate_)(
            state.use_current_model, 
            state.test_eval_state, 
            state.opponent_eval_state, 
            env_state
        )

        return state.replace(
            test_env_state = env_state,
            test_eval_state = new_model_ev_state,
            outcomes = jnp.where(
                terminated & ~state.completed,
                jnp.where(
                    state.use_current_model,
                    env_state.reward[:,1],
                    env_state.reward[:,0]
                ).flatten(),
                state.outcomes
            ),
            completed = state.completed | terminated,
            use_current_model = ~state.use_current_model,
            opponent_eval_state = best_model_ev_state
        ), None


    def test(self, 
        state: TwoPlayerTrainerState
    ) -> Tuple[TwoPlayerTrainerState, Any]:
        env_key, eval_key, new_key, op_eval_key = jax.random.split(state.key, 4)
        env_keys = jax.random.split(env_key, self.test_batch_size)
        eval_keys = jax.random.split(eval_key, self.test_batch_size)
        op_eval_keys = jax.random.split(op_eval_key, self.test_batch_size)
        

        use_current_model = jnp.zeros((self.test_batch_size,), dtype=jnp.bool_)
    

        state = TwoPlayerTestState(
            trainer_state = state.replace(key=new_key),
            test_env_state = jax.vmap(self.env.reset)(env_keys)[0],
            test_eval_state = jax.vmap(self.test_evaluator.reset)(eval_keys),
            opponent_eval_state = jax.vmap(self.test_evaluator.reset)(op_eval_keys),
            outcomes = jnp.zeros((self.test_batch_size,)),
            completed = jnp.zeros((self.test_batch_size,), dtype=jnp.bool_),
            use_current_model = use_current_model
        )

        state, _ = jax.lax.scan(
            lambda s, _: self.test_step(s),
            state,
            jnp.arange(self.config.max_episode_steps)
        )

        metrics = {
            'win_margin': jnp.sum(state.outcomes) / self.test_batch_size
        }

        return state.trainer_state, metrics
        
