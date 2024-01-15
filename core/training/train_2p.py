
from functools import partial
from typing import Optional, Tuple
import chex
from chex import dataclass
import jax
import jax.numpy as jnp

from core.evaluators.evaluator import Evaluator

from core.memory.replay_memory import EpisodeReplayBuffer
from core.training.train import CollectionState, Trainer, extract_params
from core.types import ActionMaskFn, EnvInitFn, EnvPlayerIdFn, EnvStepFn, EvalFn, ExtractModelParamsFn, TrainStepFn
from flax.training.train_state import TrainState

@dataclass(frozen=True)
class TwoPlayerTestState:
    key: jax.random.PRNGKey
    env_state: chex.ArrayTree
    cur_player_state: chex.ArrayTree
    other_player_state: chex.ArrayTree
    outcomes: float
    completed: bool

class TwoPlayerTrainer(Trainer):
    def test_step(self,
        state: TwoPlayerTestState,
        params: chex.ArrayTree,
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree, float, bool]:
        new_key, step_key = jax.random.split(state.key)
        env_state, output, reward, terminated = self._step_env_and_evaluator(
            key = step_key,
            env_state = state.env_state,
            eval_state = state.cur_player_state,
            params = params,
            is_test=True
        )
        cur_player_state = output.eval_state
        other_player_state = self.evaluator.step(state.other_player_state, output.action)

        return state.replace(
            key = new_key,
            env_state = env_state,
            cur_player_state = other_player_state,
            other_player_state = cur_player_state,
            outcomes=jnp.where(
                    (terminated & ~state.completed)[..., None],
                    reward,
                    state.outcomes
                ),
                completed = state.completed | terminated
        )
    
    def test_2p(self, 
        collection_state: CollectionState,
        params: chex.ArrayTree,
        best_model_params: chex.ArrayTree,
        num_episodes: int    
    ) -> Tuple[CollectionState, chex.ArrayTree, dict]:
        base_key, new_key = jax.random.split(collection_state.key[0], 2)
        new_cs_keys = collection_state.key.at[0].set(new_key)

        env_init_key, eval_init_key, step_key, new_key = jax.random.split(base_key, 4)
        step_keys = jax.random.split(step_key, num_episodes)
        env_init_keys = jax.random.split(env_init_key, num_episodes)
        env_state = jax.vmap(self.env_init_fn)(env_init_keys)
        eval1_key, eval2_key = jax.random.split(eval_init_key, 2)
        eval1_keys = jax.random.split(eval1_key, num_episodes)
        eval2_keys = jax.random.split(eval2_key, num_episodes)
        evaluator_init = partial(self.evaluator.init, template_embedding=self.template_env_state)
        eval_state_new = jax.vmap(evaluator_init)(eval1_keys)
        eval_state_best = jax.vmap(evaluator_init)(eval2_keys)


        step_new = partial(self.test_step, params=params)
        step_best = partial(self.test_step, params=best_model_params)

        test_state = TwoPlayerTestState(
            key = step_keys,
            env_state = env_state,
            cur_player_state = eval_state_new,
            other_player_state = eval_state_best,
            outcomes=jnp.zeros((num_episodes, 2)),
            completed=jnp.zeros((num_episodes,), dtype=jnp.bool_)
        )

        def _reset(state: TwoPlayerTestState, do_reset: bool) -> TwoPlayerTestState:
            cur_player_state = state.cur_player_state
            other_player_state = state.other_player_state
            env_state = state.env_state
            cur_player_state = jax.lax.cond(
                do_reset,
                lambda s: self.evaluator.reset(s),
                lambda s: s,
                cur_player_state
            )
            other_player_state = jax.lax.cond(
                do_reset,
                lambda s: self.evaluator.reset(s),
                lambda s: s,
                other_player_state
            )
            init_key, new_key = jax.random.split(state.key)
            env_state = jax.lax.cond(
                do_reset,
                lambda _: self.env_init_fn(init_key),
                lambda _: env_state,
                None
            )
            return state.replace(
                key=new_key,
                cur_player_state = cur_player_state,
                other_player_state = other_player_state,
                env_state = env_state
            )


        def step_step(state: TwoPlayerTestState) -> TwoPlayerTestState:
            return step_new(step_best(state))
            
        def step_while(state: TwoPlayerTestState) -> TwoPlayerTestState:
            return jax.lax.while_loop(
                lambda s: ~s.completed,
                lambda s: step_step(s),
                state
            )
        
        # make one step, reset half of environments
        # so that new params go first half the time     
        test_state = jax.vmap(step_best)(test_state)
        reset = jnp.zeros((num_episodes))
        reset = reset.at[:num_episodes // 2].set(True)
        test_state = jax.vmap(_reset)(test_state, reset)

        new_model_ids = self.env_player_id_fn(test_state.env_state)
        test_state = jax.vmap(step_while)(test_state)

        new_model_rewards = test_state.outcomes[jnp.arange(num_episodes), new_model_ids]
        best_model_rewards = test_state.outcomes[jnp.arange(num_episodes), 1 - new_model_ids]
        new_model_results = jnp.where(
            new_model_rewards > best_model_rewards,
            1,
            jnp.where(
                new_model_rewards < best_model_rewards,
                0,
                0.5
            )
        )

        win_rate = new_model_results.sum() / num_episodes

        metrics = {
            "performance_vs_best_model": win_rate
        }

        best_model_params = jax.lax.cond(
            win_rate > 0.5,
            lambda _: params,
            lambda _: best_model_params,
            None
        )

        return collection_state.replace(key=new_cs_keys), best_model_params, metrics
    
    def train_loop(self,
        collection_state: CollectionState,
        train_state: TrainState,
        warmup_steps: int,
        collection_steps_per_epoch: int,
        train_steps_per_epoch: int,
        test_episodes_per_epoch: int,
        num_epochs: int  
    ) -> Tuple[CollectionState, TrainState]:
        params = self.extract_model_params_fn(train_state)
        best_params = params
        collect_steps = jax.jit(self.collect_steps)
        warmup = jax.vmap(partial(collect_steps, params=params, num_steps=warmup_steps))
        collection_state = warmup(collection_state)

        train = jax.jit(partial(self.train_steps, num_steps=train_steps_per_epoch))
        test = jax.jit(partial(self.test_2p, num_episodes=test_episodes_per_epoch))

        for epoch in range(num_epochs):
            collection_state = jax.vmap(partial(collect_steps, params=params, num_steps=collection_steps_per_epoch))(collection_state)
            collection_state, train_state, metrics = train(collection_state, train_state)
            print(f"Epoch {epoch}: {metrics}")
            params = self.extract_model_params_fn(train_state)
            collection_state, best_params, metrics = test(collection_state, params, best_params)
            print(f"Epoch {epoch}: {metrics}")
            # TODO: checkpoints

        return collection_state, train_state