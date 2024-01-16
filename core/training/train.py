from functools import partial
from typing import Optional, Tuple
from chex import dataclass
import chex
import wandb
import jax
import jax.numpy as jnp
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.memory.replay_memory import BaseExperience, EpisodeReplayBuffer, ReplayBufferState
from core.types import ActionMaskFn, EnvInitFn, EnvPlayerIdFn, EnvStepFn, EvalFn, ExtractModelParamsFn, TrainStepFn
from flax.training.train_state import TrainState

@dataclass(frozen=True)
class CollectionState:
    key: jax.random.PRNGKey
    eval_state: chex.ArrayTree
    env_state: chex.ArrayTree
    buffer_state: ReplayBufferState


@dataclass(frozen=True)
class TestState:
    key: jax.random.PRNGKey
    eval_state: chex.ArrayTree
    env_state: chex.ArrayTree
    outcomes: chex.Array
    completed: chex.Array

def extract_params(state: TrainState) -> chex.ArrayTree:
    if hasattr(state, 'batch_stats'):
        return {'params': state.params, 'batch_stats': state.batch_stats}
    return {'params': state.params}


class Trainer:
    def __init__(self,
        train_batch_size: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        eval_fn: EvalFn,
        env_player_id_fn: EnvPlayerIdFn,
        action_mask_fn: ActionMaskFn,
        train_step_fn: TrainStepFn,
        evaluator: Evaluator,
        memory_buffer: EpisodeReplayBuffer,
        template_env_state: chex.ArrayTree,
        evaluator_kwargs_train: dict,
        evaluator_kwargs_test: Optional[dict] = None,
        extract_model_params_fn: Optional[ExtractModelParamsFn] = extract_params,
        wandb_project_name: str = "",
    ):
        self.train_batch_size = train_batch_size
        self.evaluator = evaluator
        self.memory_buffer = memory_buffer
        self.env_step_fn = env_step_fn
        self.env_init_fn = env_init_fn
        self.env_player_id_fn = env_player_id_fn
        self.eval_fn = eval_fn
        self.action_mask_fn = action_mask_fn
        self.train_step_fn = train_step_fn
        self.extract_model_params_fn = extract_model_params_fn
        self.template_env_state = template_env_state

        evaluate = partial(self.evaluator.evaluate, 
            env_step_fn=self.env_step_fn, 
            env_player_id_fn=self.env_player_id_fn,
            eval_fn=self.eval_fn, 
            action_mask_fn=self.action_mask_fn)
        
        self.evaluator_kwargs_train = evaluator_kwargs_train
        self.evaluate_train = partial(evaluate, **self.evaluator_kwargs_train)
        if evaluator_kwargs_test is None:
            self.evaluator_kwargs_test = evaluator_kwargs_train
            self.evaluate_test = self.evaluate_train
        else:
            self.evaluator_kwargs_test = evaluator_kwargs_test
            self.evaluate_test = partial(evaluate, **self.evaluator_kwargs_test)
        
        self.use_wandb = wandb_project_name != ""
        if self.use_wandb:
            self.run = wandb.init(
            # Set the project where this run will be logged
                project=wandb_project_name,
                # Track hyperparameters and run metadata
                config={},
            )

    def _step_env_and_evaluator(self,
        key: jax.random.PRNGKey,
        env_state: chex.ArrayTree,
        eval_state: chex.ArrayTree,
        params: chex.ArrayTree,
        is_test: bool,
    ) -> Tuple[chex.ArrayTree, EvalOutput, float, bool]:
        evaluate_fn = self.evaluate_test if is_test else self.evaluate_train
        output = evaluate_fn(eval_state=eval_state, env_state=env_state, params=params)
        eval_state = output.eval_state
        env_state, reward, terminated = self.env_step_fn(env_state, output.action)
        eval_state = jax.lax.cond(
            terminated,
            lambda s: self.evaluator.reset(s),
            lambda s: self.evaluator.step(s, output.action),
            eval_state
        )
        env_state = jax.lax.cond(
            terminated,
            lambda _: self.env_init_fn(key),
            lambda _: env_state,
            None
        )
        output = output.replace(eval_state=eval_state)
        return env_state, output, reward, terminated    
    
    def collect(self,
        state: CollectionState,
        params: chex.ArrayTree
    ) -> CollectionState:
        step_key, new_key = jax.random.split(state.key)
        new_env_state, eval_output, reward, terminated = \
            self._step_env_and_evaluator(
                key = step_key,
                env_state = state.env_state,
                eval_state = state.eval_state,
                params = params,
                is_test = False
            )
        
        buffer_state = self.memory_buffer.add_experience(
            state = state.buffer_state,
            experience = BaseExperience(
                env_state=state.env_state,
                policy_mask=self.action_mask_fn(state.env_state),
                policy_weights=eval_output.policy_weights,
                reward=jnp.empty_like(state.buffer_state.buffer.reward[0])
            )
        )
        
        buffer_state = jax.lax.cond(
            terminated,
            lambda s: self.memory_buffer.assign_rewards(s, reward),
            lambda s: s,
            buffer_state
        )

        return state.replace(
            key=new_key,
            eval_state=eval_output.eval_state,
            env_state=new_env_state,
            buffer_state=buffer_state
        )

    
    def collect_steps(self,
        state: CollectionState,
        params: chex.ArrayTree,
        num_steps: int
    ) -> CollectionState:
        collect = partial(self.collect, params=params)
        return jax.lax.fori_loop(
            0, num_steps, 
            lambda _, s: collect(s), 
            state
        )
    
    def train_steps(self,
        collection_state: CollectionState,
        train_state: TrainState,
        num_steps: int
    ) -> Tuple[CollectionState, TrainState, dict]:
        buffer_state = collection_state.buffer_state
        

        def sample_and_train(ts: TrainState, key: jax.random.PRNGKey) -> TrainState:
            samples = self.memory_buffer.sample_across_batches(buffer_state, key, sample_size=self.train_batch_size)
            return self.train_step_fn(samples, ts)
        
        sample_key, new_key = jax.random.split(collection_state.key[0], 2)
        new_cs_keys = collection_state.key.at[0].set(new_key)
        sample_keys = jax.random.split(sample_key, num_steps)

        train_state, metrics = jax.lax.scan(
            sample_and_train,
            train_state,
            xs=sample_keys,
        )
        mean_metrics = jax.tree_map(lambda x: x.mean(), metrics)
        
        return collection_state.replace(key=new_cs_keys), train_state, mean_metrics 
    
    def test(self,
        collection_state: CollectionState,
        params: chex.ArrayTree,
        num_episodes: int
    ) -> Tuple[CollectionState, dict]:
        base_key, new_key = jax.random.split(collection_state.key[0], 2)
        new_cs_keys = collection_state.key.at[0].set(new_key)
        # init a new test env state w/ batch size = num_episodes
        env_init_key, eval_init_key, step_key, new_key = jax.random.split(base_key, 4)
        env_init_keys = jax.random.split(env_init_key, num_episodes)
        eval_init_keys = jax.random.split(eval_init_key, num_episodes)
        env_state = jax.vmap(self.env_init_fn)(env_init_keys)
        evaluator_init = partial(self.evaluator.init, template_embedding=self.template_env_state)
        eval_state = jax.vmap(evaluator_init)(eval_init_keys)

        test_state = TestState(
            key=step_key,
            eval_state=eval_state,
            env_state=env_state,
            outcomes=jnp.zeros((num_episodes,)),
            completed=jnp.zeros((num_episodes,), dtype=jnp.bool_)
        )

        def test_step(state: TestState) -> TestState:
            new_key, step_key = jax.random.split(state.key)
            env_state, eval_output, reward, terminated = \
                self._step_env_and_evaluator(
                    self.evaluator,
                    step_key,
                    env_state = state.env_state,
                    eval_state = state.eval_state,
                    params = params,
                    is_test = True
                )
            return state.replace(
                key=new_key,
                eval_state=eval_output.eval_state,
                env_state=env_state,
                outcomes=jnp.where(
                    (terminated & ~state.completed)[..., None],
                    reward,
                    state.outcomes
                ),
                completed = state.completed | terminated
            )
        
        def step_while(state: TestState) -> TestState:
            return jax.lax.while_loop(
                lambda s: ~s.completed,
                lambda s: test_step(s),
                state
            )
        
        test_state = jax.vmap(step_while)(test_state)
        
        # ok now we can return some metrics or something
        metrics = {
            "avg_test_reward": test_state.outcomes.mean(),
        }

        return collection_state.replace(key=new_cs_keys), metrics
    
    def log_metrics(self, metrics: dict, epoch: int):
        print(f"Epoch {epoch}: {metrics}")
        if self.use_wandb:
            wandb.log(metrics)
    
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
        collect_steps = jax.jit(self.collect_steps)
        warmup = jax.vmap(partial(collect_steps, params=params, num_steps=warmup_steps))
        collection_state = warmup(collection_state)

        train = jax.jit(partial(self.train_steps, num_steps=train_steps_per_epoch))
        test = jax.jit(partial(self.test, num_episodes=test_episodes_per_epoch))

        for epoch in range(num_epochs):
            collection_state = jax.vmap(partial(collect_steps, params=params, num_steps=collection_steps_per_epoch))(collection_state)
            collection_state, train_state, metrics = train(collection_state, train_state)
            self.log_metrics(metrics, epoch)
            params = self.extract_model_params_fn(train_state)
            collection_state, metrics = test(collection_state, params)
            self.log_metrics(metrics, epoch)
            # TODO: checkpoints

        return collection_state, train_state

    