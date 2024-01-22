from functools import partial
import os
import shutil
from typing import Any, Optional, Tuple
from chex import dataclass
import chex
import wandb
import jax
import jax.numpy as jnp
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.memory.replay_memory import BaseExperience, EpisodeReplayBuffer, ReplayBufferState
from core.types import EnvInitFn, EnvStepFn, EvalFn, ExtractModelParamsFn, StepMetadata, TrainStepFn
from flax.training.train_state import TrainState
import orbax
from flax.training import orbax_utils

@dataclass(frozen=True)
class CollectionState:
    key: jax.random.PRNGKey
    eval_state: chex.ArrayTree
    env_state: chex.ArrayTree
    buffer_state: ReplayBufferState
    metadata: StepMetadata

@dataclass(frozen=True)
class TestState:
    key: jax.random.PRNGKey
    eval_state: chex.ArrayTree
    env_state: chex.ArrayTree
    outcomes: chex.Array
    completed: chex.Array
    metadata: StepMetadata

@dataclass(frozen=True)
class TrainLoopOutput:
    collection_state: CollectionState
    train_state: TrainState
    best_params: Optional[chex.ArrayTree]

def extract_params(state: TrainState) -> chex.ArrayTree:
    if hasattr(state, 'batch_stats'):
        return {'params': state.params, 'batch_stats': state.batch_stats}
    return {'params': state.params}


class Trainer:
    def __init__(self,
        train_batch_size: int,
        evaluator: Evaluator,        
        memory_buffer: EpisodeReplayBuffer,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        eval_fn: EvalFn,
        train_step_fn: TrainStepFn,
        evaluator_test: Optional[Evaluator] = None,
        extract_model_params_fn: Optional[ExtractModelParamsFn] = extract_params,
        wandb_project_name: str = "",
        ckpt_dir: str = "/tmp/turbozero_checkpoints",
        max_checkpoints: int = 2
    ):
        self.train_batch_size = train_batch_size
        self.evaluator_train = evaluator
        self.evaluator_test = evaluator_test if evaluator_test is not None else evaluator
        self.memory_buffer = memory_buffer
        self.env_step_fn = env_step_fn
        self.env_init_fn = env_init_fn
        self.eval_fn = eval_fn
        self.train_step_fn = train_step_fn
        self.extract_model_params_fn = extract_model_params_fn
        
        self.ckpt_dir = ckpt_dir
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)  

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_checkpoints, create=True)
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            ckpt_dir, orbax_checkpointer, options)

        self.evaluate_fn_train = partial(self.evaluator_train.evaluate, 
            env_step_fn=self.env_step_fn,
            eval_fn=self.eval_fn)
        
        self.evaluate_fn_test = partial(self.evaluator_test.evaluate, 
            env_step_fn=self.env_step_fn,
            eval_fn=self.eval_fn)
        
        self.wandb_project_name = wandb_project_name
        
        self.template_env_state = self.make_template_env_state()

    def get_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'evaluator_train': self.evaluator_train.__class__.__name__,
            'evaluator_train_config': self.evaluator_train.get_config(),
            'evaluator_test': self.evaluator_test.__class__.__name__,
            'evaluator_test_config': self.evaluator_test.get_config(),
            'memory_buffer': self.memory_buffer.__class__.__name__,
            'memory_buffer_config': self.memory_buffer.get_config(),
        }

    def _step_env_and_evaluator(self,
        key: jax.random.PRNGKey,
        env_state: chex.ArrayTree,
        eval_state: chex.ArrayTree,
        metadata: StepMetadata,
        params: chex.ArrayTree,
        is_test: bool,
    ) -> Tuple[chex.ArrayTree, EvalOutput, StepMetadata, bool, chex.Array]:
        evaluate_fn = self.evaluate_fn_test if is_test else self.evaluate_fn_train
        evaluator = self.evaluator_test if is_test else self.evaluator_train
        output = evaluate_fn(eval_state=eval_state, env_state=env_state, root_metadata=metadata, params=params)
        eval_state = output.eval_state
        env_state, metadata = self.env_step_fn(env_state, output.action)
        terminated = metadata.terminated
        rewards = metadata.rewards
        eval_state = jax.lax.cond(
            terminated,
            lambda s: evaluator.reset(s),
            lambda s: evaluator.step(s, output.action),
            eval_state
        )
        env_state, metadata = jax.lax.cond(
            terminated,
            lambda _: self.env_init_fn(key),
            lambda _: (env_state, metadata),
            None
        )
        output = output.replace(eval_state=eval_state)
        return env_state, output, metadata, terminated, rewards
    
    def collect(self,
        state: CollectionState,
        params: chex.ArrayTree
    ) -> CollectionState:
        step_key, new_key = jax.random.split(state.key)
        new_env_state, eval_output, new_metadata, terminated, rewards = \
            self._step_env_and_evaluator(
                key = step_key,
                env_state = state.env_state,
                eval_state = state.eval_state,
                metadata = state.metadata,
                params = params,
                is_test = False
            )
        
        buffer_state = self.memory_buffer.add_experience(
            state = state.buffer_state,
            experience = BaseExperience(
                env_state=state.env_state,
                policy_mask=state.metadata.action_mask,
                policy_weights=eval_output.policy_weights,
                reward=jnp.empty_like(state.metadata.rewards)
            )
        )
        
        buffer_state = jax.lax.cond(
            terminated,
            lambda s: self.memory_buffer.assign_rewards(s, rewards),
            lambda s: s,
            buffer_state
        )

        return state.replace(
            key=new_key,
            eval_state=eval_output.eval_state,
            env_state=new_env_state,
            buffer_state=buffer_state,
            metadata=new_metadata
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
        num_episodes: int,
        best_params: Optional[chex.ArrayTree] = None
    ) -> Tuple[CollectionState, dict]:
        base_key, new_key = jax.random.split(collection_state.key[0], 2)
        new_cs_keys = collection_state.key.at[0].set(new_key)
        # init a new test env state w/ batch size = num_episodes
        env_init_key, eval_init_key, step_key, new_key = jax.random.split(base_key, 4)
        env_init_keys = jax.random.split(env_init_key, num_episodes)
        eval_init_keys = jax.random.split(eval_init_key, num_episodes)
        env_state, metadata = jax.vmap(self.env_init_fn)(env_init_keys)
        evaluator_init = partial(self.evaluator_test.init, template_embedding=self.template_env_state)
        eval_state = jax.vmap(evaluator_init)(eval_init_keys)

        test_state = TestState(
            key=step_key,
            eval_state=eval_state,
            env_state=env_state,
            outcomes=jnp.zeros((num_episodes,)),
            completed=jnp.zeros((num_episodes,), dtype=jnp.bool_),
            metadata=metadata
        )

        def test_step(state: TestState) -> TestState:
            new_key, step_key = jax.random.split(state.key)
            env_state, eval_output, metadata, terminated, rewards = \
                self._step_env_and_evaluator(
                    self.evaluator_test,
                    step_key,
                    env_state = state.env_state,
                    eval_state = state.eval_state,
                    metadata = state.metadata,
                    params = params,
                    is_test = True
                )
            return state.replace(
                key=new_key,
                eval_state=eval_output.eval_state,
                env_state=env_state,
                outcomes=jnp.where(
                    (terminated & ~state.completed)[..., None],
                    rewards,
                    state.outcomes
                ),
                completed = state.completed | terminated,
                metadata = metadata
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

        return collection_state.replace(key=new_cs_keys), metrics, best_params
    
    def log_metrics(self, metrics: dict, epoch: int, step: Optional[int] = None):
        print(f"Epoch {epoch}: {metrics}")
        if self.use_wandb:
            wandb.log(metrics, step)

    def save_checkpoint(self, train_state: TrainState, epoch: int, **kwargs):
        ckpt = {'train_state': train_state}
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
        collection_state: Optional[CollectionState] = None,
        wandb_run: Optional[Any] = None,
        extra_wandb_config: Optional[dict] = {}
    ) -> Tuple[CollectionState, TrainState]:
        if wandb_run is None and self.wandb_project_name != "":
            self.run = wandb.init(
            # Set the project where this run will be logged
                project=self.wandb_project_name,
                # Track hyperparameters and run metadata
                config={
                    'collection_batch_size': batch_size,
                    'warmup_steps': warmup_steps,
                    'collection_steps_per_epoch': collection_steps_per_epoch,
                    'train_steps_per_epoch': train_steps_per_epoch,
                    'test_episodes_per_epoch': test_episodes_per_epoch,
                    **self.get_config(), **extra_wandb_config
                },
            )


        if collection_state is None:
            collection_state = self.init_collection_state(key, batch_size)

        params = self.extract_model_params_fn(train_state)
        collect_steps = jax.jit(self.collect_steps)
        warmup = jax.vmap(partial(collect_steps, params=params, num_steps=warmup_steps))
        collection_state = warmup(collection_state)

        train = jax.jit(partial(self.train_steps, num_steps=train_steps_per_epoch))
        test = jax.jit(partial(self.test, num_episodes=test_episodes_per_epoch))

        collection_batch_size = collection_state.key.shape[0]
        while cur_epoch < num_epochs:
            cur_epoch += 1
            collection_steps = collection_batch_size * cur_epoch * collection_steps_per_epoch
            collection_state = jax.vmap(partial(collect_steps, params=params, num_steps=collection_steps_per_epoch))(collection_state)
            collection_state, train_state, metrics = train(collection_state, train_state)
            self.log_metrics(metrics, cur_epoch, step=collection_steps)
            params = self.extract_model_params_fn(train_state)
            collection_state, metrics, best_params = test(collection_state, params, best_params=best_params)
            self.log_metrics(metrics, cur_epoch, step=collection_steps)
            self.save_checkpoint(train_state, cur_epoch, best_params)
    
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            best_params=best_params
        )

    
    
    def make_template_env_state(self) -> chex.ArrayTree:
        env_state, _ = self.env_init_fn(jax.random.PRNGKey(0))
        return env_state
    
    
    def make_template_experience(self) -> BaseExperience:
        env_state, metadata = self.env_init_fn(jax.random.PRNGKey(0))
        return BaseExperience(
            env_state=env_state,
            policy_mask=metadata.action_mask,
            policy_weights=jnp.zeros_like(metadata.action_mask, dtype=jnp.float32),
            reward=jnp.zeros_like(metadata.rewards)
        )
    

    def init_collection_state(self, key: jax.random.PRNGKey, batch_size: int, template_experience: Optional[BaseExperience] = None):
        if template_experience is None:
            template_experience = self.make_template_experience()
        buff_key, key = jax.random.split(key)
        buffer_state = self.memory_buffer.init_batched_buffer(buff_key, batch_size, template_experience)
        env_init_key, key = jax.random.split(key)
        env_keys = jax.random.split(env_init_key, batch_size)
        env_state, metadata = jax.vmap(self.env_init_fn)(env_keys)
        eval_key, key = jax.random.split(key)
        eval_keys = jax.random.split(eval_key, batch_size)
        evaluator_init = partial(self.evaluator_train.init, template_embedding=self.template_env_state)
        eval_state = jax.vmap(evaluator_init)(eval_keys)
        state_keys = jax.random.split(key, batch_size)
        return CollectionState(
            key=state_keys,
            eval_state=eval_state,
            env_state=env_state,
            buffer_state=buffer_state,
            metadata=metadata
        )

    