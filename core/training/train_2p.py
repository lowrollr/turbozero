
from functools import partial
from typing import Optional, Tuple
import chex
from chex import dataclass
import jax
import jax.numpy as jnp

from core.training.train import CollectionState, Trainer
from flax.training.train_state import TrainState
from flax.training import orbax_utils

from core.types import StepMetadata

@dataclass(frozen=True)
class TwoPlayerTestState:
    key: jax.random.PRNGKey
    env_state: chex.ArrayTree
    cur_player_state: chex.ArrayTree
    other_player_state: chex.ArrayTree
    outcomes: float
    completed: bool
    metadata: StepMetadata

class TwoPlayerTrainer(Trainer):
    def test_step(self,
        state: TwoPlayerTestState,
        params: chex.ArrayTree,
    ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree, float, bool]:
        new_key, step_key = jax.random.split(state.key)
        env_state, output, metadata, terminated, rewards = self._step_env_and_evaluator(
            key = step_key,
            env_state = state.env_state,
            eval_state = state.cur_player_state,
            metadata=state.metadata,
            params = params,
            is_test=True
        )
        cur_player_state = output.eval_state
        other_player_state = self.evaluator_test.step(state.other_player_state, output.action)

        return state.replace(
            key = new_key,
            env_state = env_state,
            cur_player_state = other_player_state,
            other_player_state = cur_player_state,
            outcomes=jnp.where(
                (terminated & ~state.completed)[..., None],
                rewards,
                state.outcomes
            ),
            completed = state.completed | terminated,
            metadata = metadata
        )
    
    def test(self, 
        collection_state: CollectionState,
        params: chex.ArrayTree,
        num_episodes: int,    
        best_params: chex.ArrayTree
    ) -> Tuple[CollectionState, chex.ArrayTree, dict]:
        base_key, new_key = jax.random.split(collection_state.key[0], 2)
        new_cs_keys = collection_state.key.at[0].set(new_key)

        env_init_key, eval_init_key, step_key, new_key = jax.random.split(base_key, 4)
        step_keys = jax.random.split(step_key, num_episodes)
        env_init_keys = jax.random.split(env_init_key, num_episodes)
        env_state, metadata = jax.vmap(self.env_init_fn)(env_init_keys)
        eval1_key, eval2_key = jax.random.split(eval_init_key, 2)
        eval1_keys = jax.random.split(eval1_key, num_episodes)
        eval2_keys = jax.random.split(eval2_key, num_episodes)
        evaluator_init = partial(self.evaluator_test.init, template_embedding=self.template_env_state)
        eval_state_new = jax.vmap(evaluator_init)(eval1_keys)
        eval_state_best = jax.vmap(evaluator_init)(eval2_keys)


        step_new = partial(self.test_step, params=params)
        step_best = partial(self.test_step, params=best_params)

        test_state = TwoPlayerTestState(
            key = step_keys,
            env_state = env_state,
            cur_player_state = eval_state_new,
            other_player_state = eval_state_best,
            outcomes=jnp.zeros((num_episodes, 2)),
            completed=jnp.zeros((num_episodes,), dtype=jnp.bool_),
            metadata = metadata
        )

        def _reset(state: TwoPlayerTestState, do_reset: bool) -> TwoPlayerTestState:
            cur_player_state = state.cur_player_state
            other_player_state = state.other_player_state
            env_state = state.env_state
            cur_player_state = jax.lax.cond(
                do_reset,
                lambda s: self.evaluator_test.reset(s),
                lambda s: s,
                cur_player_state
            )
            other_player_state = jax.lax.cond(
                do_reset,
                lambda s: self.evaluator_test.reset(s),
                lambda s: s,
                other_player_state
            )
            init_key, new_key = jax.random.split(state.key)
            env_state, metadata = jax.lax.cond(
                do_reset,
                lambda _: self.env_init_fn(init_key),
                lambda _: (env_state, state.metadata),
                None
            )
            return state.replace(
                key=new_key,
                cur_player_state = cur_player_state,
                other_player_state = other_player_state,
                env_state = env_state,
                metadata = metadata
            )


        def step_step(state: TwoPlayerTestState) -> TwoPlayerTestState:
            return step_best(step_new(state))
            
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

        new_model_ids = test_state.metadata.cur_player_id
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
        win_rate_going_first = new_model_results[:num_episodes // 2].sum() / (num_episodes // 2)
        win_rate_going_second = new_model_results[num_episodes // 2:].sum() / (num_episodes - (num_episodes // 2))
        win_rate = new_model_results.sum() / num_episodes

        metrics = {
            "performance_going_first_vs_best": win_rate_going_first,
            "performance_going_second_vs_best": win_rate_going_second,
            "performance_vs_best": win_rate
        }

        best_params = jax.lax.cond(
            win_rate > 0.5,
            lambda _: params,
            lambda _: best_params,
            None
        )

        return collection_state.replace(key=new_cs_keys), metrics, best_params
    
    def save_checkpoint(self, train_state: TrainState, epoch: int, best_params: chex.ArrayTree):
        ckpt = {'train_state': train_state, 'best_params': best_params}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})
    
    def train_loop(self,
        key: jax.random.PRNGKey,
        batch_size: int,
        train_state: TrainState,
        warmup_steps: int,
        collection_steps_per_epoch: int,
        train_steps_per_epoch: int,
        test_episodes_per_epoch: int,
        num_epochs: int,
        cur_epoch: int = 0,
        best_params: Optional[chex.ArrayTree] = None,
        collection_state: Optional[CollectionState] = None
    ) -> Tuple[CollectionState, TrainState]:
        if best_params is None:
            best_params = self.extract_model_params_fn(train_state)
        return super().train_loop(
            key=key,
            batch_size=batch_size,
            train_state=train_state,
            warmup_steps=warmup_steps,
            collection_steps_per_epoch=collection_steps_per_epoch,
            train_steps_per_epoch=train_steps_per_epoch,
            test_episodes_per_epoch=test_episodes_per_epoch,
            num_epochs=num_epochs,
            cur_epoch=cur_epoch,
            best_params=best_params,
            collection_state=collection_state 
        )
    