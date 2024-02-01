from functools import partial
import os
import shutil
from typing import Any, List, Optional, Tuple
from chex import dataclass
import chex
import wandb
import jax
import jax.numpy as jnp
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.memory.replay_memory import BaseExperience, EpisodeReplayBuffer, ReplayBufferState
from core.common import step_env_and_evaluator
from core.testing.tester import BaseTester, TestState
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
class TrainLoopOutput:
    collection_state: CollectionState
    train_state: TrainState
    test_states: List[TestState]
    cur_epoch: int

def extract_params(state: TrainState) -> chex.ArrayTree:
    if hasattr(state, 'batch_stats'):
        return {'params': state.params, 'batch_stats': state.batch_stats}
    return {'params': state.params}


class Trainer:
    def __init__(self,
        train_batch_size: int,
        evaluator: Evaluator,        
        memory_buffer: EpisodeReplayBuffer,
        testers: List[BaseTester],
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
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
        self.train_step_fn = train_step_fn
        self.extract_model_params_fn = extract_model_params_fn
        self.testers = testers

        self.ckpt_dir = ckpt_dir
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)  

        self.step_train = partial(step_env_and_evaluator,
            evaluator=self.evaluator_train,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn
        )
        
        self.step_test = partial(step_env_and_evaluator,
            evaluator=self.evaluator_test,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn
        )
        
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_checkpoints, create=True)
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            ckpt_dir, orbax_checkpointer, options)

        self.wandb_project_name = wandb_project_name
        self.use_wandb = wandb_project_name != ""
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
    
    def collect(self,
        state: CollectionState,
        params: chex.ArrayTree
    ) -> CollectionState:
        step_key, new_key = jax.random.split(state.key)
        eval_output, new_env_state, new_metadata, terminated, rewards = \
            self.step_train(
                key = step_key,
                env_state = state.env_state,
                env_state_metadata = state.metadata,
                eval_state = state.eval_state,
                params = params
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

    @partial(jax.jit, static_argnums=(0, 3))
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
    
    @partial(jax.jit, static_argnums=(0, 3))
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
    
    def log_metrics(self, metrics: dict, epoch: int, step: Optional[int] = None):
        print(f"Epoch {epoch}: {metrics}")
        if self.use_wandb:
            wandb.log(metrics, step)

    def save_checkpoint(self, train_state: TrainState, epoch: int, **kwargs):
        ckpt = {'train_state': train_state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(epoch, ckpt, save_kwargs={'save_args': save_args})

    def load_train_state_from_checkpoint(self, path_to_checkpoint: str) -> TrainState:
        ckpt = self.checkpoint_manager.restore(path_to_checkpoint)
        return ckpt['train_state']
    
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

    def train_loop(self,
        key: jax.random.PRNGKey,
        batch_size: int,
        train_state: TrainState,
        warmup_steps: int,
        collection_steps_per_epoch: int,
        train_steps_per_epoch: int,
        num_epochs: int,
        cur_epoch: int = 0,
        collection_state: Optional[CollectionState] = None,
        test_states: List[chex.ArrayTree] = None, 
        wandb_run: Optional[Any] = None,
        extra_wandb_config: Optional[dict] = None
    ) -> Tuple[CollectionState, TrainState]:
        if test_states is None:
            test_states = []
        if extra_wandb_config is None:
            extra_wandb_config = {}
        
        if wandb_run is None and self.use_wandb:
            self.run = wandb.init(
            # Set the project where this run will be logged
                project=self.wandb_project_name,
                # Track hyperparameters and run metadata
                config={
                    'collection_batch_size': batch_size,
                    'warmup_steps': warmup_steps,
                    'collection_steps_per_epoch': collection_steps_per_epoch,
                    'train_steps_per_epoch': train_steps_per_epoch,
                    **self.get_config(), **extra_wandb_config
                },
            )

        if not test_states:
            for tester in self.testers:
                tester_init_key, key = jax.random.split(key)
                test_states.append(tester.init(tester_init_key, params=self.extract_model_params_fn(train_state)))

        if collection_state is None:
            init_key, key = jax.random.split(key)
            collection_state = self.init_collection_state(init_key, batch_size)
        collection_batch_size = collection_state.key.shape[0]

        params = self.extract_model_params_fn(train_state)
        warmup = jax.vmap(partial(self.collect_steps, params=params, num_steps=warmup_steps))
        collection_state = warmup(collection_state)

        train = partial(self.train_steps, num_steps=train_steps_per_epoch)
        test_fns = [
            partial(tester.run, env_step_fn=self.env_step_fn, env_init_fn=self.env_init_fn, evaluator=self.evaluator_test)
            for tester in self.testers
        ]

        while cur_epoch < num_epochs:
            collection_steps = collection_batch_size * (cur_epoch+1) * collection_steps_per_epoch
            collection_state = jax.vmap(partial(self.collect_steps, params=params, num_steps=collection_steps_per_epoch))(collection_state)
            collection_state, train_state, metrics = train(collection_state, train_state)
            self.log_metrics(metrics, cur_epoch, step=collection_steps)
            params = self.extract_model_params_fn(train_state)
            
            for i, (test_fn, test_state) in enumerate(zip(test_fns, test_states)):
                new_test_state, metrics = test_fn(epoch_num=cur_epoch, params=params, state=test_state)
                self.log_metrics(metrics, cur_epoch, step=collection_steps)
                test_states[i] = new_test_state

            cur_epoch += 1
            self.save_checkpoint(train_state, cur_epoch)
    
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            test_states=test_states,
            cur_epoch=cur_epoch
        )
    
    