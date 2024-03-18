from functools import partial
import os
import shutil
from typing import Any, List, Optional, Tuple
from chex import dataclass
import chex
import flax
import optax
import wandb
import jax
import jax.numpy as jnp
from core.evaluators.evaluator import Evaluator
from core.memory.replay_memory import BaseExperience, EpisodeReplayBuffer, ReplayBufferState
from core.common import partition, step_env_and_evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn, ExtractModelParamsFn, LossFn, StateToNNInputFn, StepMetadata
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


class TrainStateWithBS(TrainState):
    batch_stats: chex.ArrayTree
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

def extract_params(state: TrainState) -> chex.ArrayTree:
    if hasattr(state, 'batch_stats'):
        return {'params': state.params, 'batch_stats': state.batch_stats}
    return {'params': state.params}

def copy_for_devices(state: TrainState, num_devices: int) -> TrainState:
    if hasattr(state, 'batch_stats'):
        return state.replace(
            params = jax.tree_map(lambda x: jnp.array([x] * num_devices), state.params),
            batch_stats = jax.tree_map(lambda x: jnp.array([x] * num_devices), state.batch_stats)
        )
    return state.replace(
        params = jax.tree_map(lambda x: jnp.array([x] * num_devices), state.params)
    )

class Trainer:
    def __init__(self,
        train_batch_size: int,
        evaluator: Evaluator,        
        memory_buffer: EpisodeReplayBuffer,
        testers: List[BaseTester],
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        state_to_nn_input_fn: StateToNNInputFn,
        nn: flax.linen.Module,
        loss_fn: LossFn,
        optimizer: optax.GradientTransformation,
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

        self.state_to_nn_input_fn = state_to_nn_input_fn
        self.nn = nn
        self.loss_fn = loss_fn
        self.optimizer = optimizer

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
        

    def init_train_state(self, key: jax.random.PRNGKey) -> TrainState:
        sample_env_state = self.make_template_env_state()
        sample_obs = self.state_to_nn_input_fn(sample_env_state)
        variables = self.nn.init(key, sample_obs[None, ...], train=False)
        params = variables['params']
        if 'batch_stats' in variables:
            return TrainStateWithBS.create(
                apply_fn=self.nn.apply,
                params=params,
                tx=self.optimizer,
                batch_stats=variables['batch_stats']
            )
    
        return TrainState.create(
            apply_fn=self.nn.apply,
            params=params,
            tx=self.optimizer,
        )

        
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
                observation_nn=self.state_to_nn_input_fn(new_env_state),
                policy_mask=state.metadata.action_mask,
                policy_weights=eval_output.policy_weights,
                reward=jnp.empty_like(state.metadata.rewards),
                cur_player_id=state.metadata.cur_player_id
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

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 3))
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
        
        def one_train_step(ts: TrainState, key: chex.PRNGKey) -> Tuple[TrainState, dict]:
            samples = self.memory_buffer.sample(buffer_state, key, sample_size=self.train_batch_size)
            # re-partition samples across available devices
            # partitioned_samples = partition(samples, jax.local_device_count())
            grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
            (loss, (metrics, updates)), grads = grad_fn(ts.params, ts, samples)
            grads = jax.lax.pmean(grads, axis_name='d')
            ts = ts.apply_gradients(grads=grads)
            if hasattr(ts, 'batch_stats'):
                ts = ts.replace(batch_stats=jax.lax.pmean(updates['batch_stats'], axis_name='d'))
            metrics = {
                **metrics,
                'loss': loss
            }
            return ts, metrics
        
        @partial(jax.pmap, axis_name='d')
        def train(ts: TrainState, key: chex.PRNGKey) -> Tuple[TrainState, dict]:
            keys = jax.random.split(key, num_steps)
            ts, metrics = jax.lax.scan(
                one_train_step,
                ts,
                xs=keys,
            )
            # mean across steps
            mean_metrics = jax.tree_map(lambda x: x.mean(), metrics)
            return ts, mean_metrics
        
        sample_key, new_key = jax.random.split(collection_state.key[0,0], 2)
        new_keys = collection_state.key.at[0,0].set(new_key)
        sample_keys = jax.random.split(sample_key, jax.local_device_count())
        train_state, metrics = train(train_state, sample_keys)
        # mean across devices
        metrics = jax.tree_map(lambda x: x.mean(), metrics)
        return collection_state.replace(key=new_keys), train_state, metrics
    
    
    def log_metrics(self, metrics: dict, epoch: int, step: Optional[int] = None):
        metrics = {k: f"{v.item():.4f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {metrics}")
        if self.use_wandb:
            wandb.log(metrics, step)

    def save_checkpoint(self, train_state: TrainState, epoch: int, **kwargs):
        ckpt = {'train_state': flax.jax_utils.unreplicate(train_state)}
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
            observation_nn=self.state_to_nn_input_fn(env_state),
            policy_mask=metadata.action_mask,
            policy_weights=jnp.zeros_like(metadata.action_mask, dtype=jnp.float32),
            reward=jnp.zeros_like(metadata.rewards),
            cur_player_id=metadata.cur_player_id
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
        
        warmup_steps: int,
        collection_steps_per_epoch: int,
        train_steps_per_epoch: int,
        num_epochs: int,
        cur_epoch: int = 0,
        train_state: Optional[TrainState] = None,
        collection_state: Optional[CollectionState] = None,
        test_states: List[chex.ArrayTree] = [], 
        wandb_run: Optional[Any] = None,
        extra_wandb_config: Optional[dict] = None,
        num_devices: Optional[int] = None
    ) -> Tuple[CollectionState, TrainState]:
        if num_devices is None:
            num_devices = jax.local_device_count()
        if batch_size % num_devices != 0:
            raise ValueError(f"Batch size must be divisible by the number of devices. Got {batch_size} batch size and {num_devices} devices.")
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

        if train_state is None:
            train_key, key = jax.random.split(key)
            train_state = self.init_train_state(train_key)
            train_state = flax.jax_utils.replicate(train_state)

        if not test_states:
            for tester in self.testers:
                tester_init_key, key = jax.random.split(key)
                init_keys = jax.random.split(tester_init_key, num_devices)
                state = tester.init(init_keys, params=self.extract_model_params_fn(train_state))
                test_states.append(state)
        
        if collection_state is None:
            init_key, key = jax.random.split(key)
            collection_state = partition(self.init_collection_state(init_key, batch_size), num_devices)


        params = self.extract_model_params_fn(train_state)
        collect = jax.vmap(self.collect_steps, in_axes=(1, None, None), out_axes=1)
        collection_state = collect(collection_state, params, warmup_steps)


        while cur_epoch < num_epochs:
            collection_steps = batch_size * (cur_epoch+1) * collection_steps_per_epoch
            collection_state = collect(collection_state, params, collection_steps_per_epoch)
            collection_state, train_state, metrics = self.train_steps(collection_state, train_state, train_steps_per_epoch)
            self.log_metrics(metrics, cur_epoch, step=collection_steps)
            params = self.extract_model_params_fn(train_state)
            
            for i, test_state in enumerate(test_states):
                new_test_state, metrics = self.testers[i].run(
                    cur_epoch, self.env_step_fn, self.env_init_fn, 
                    self.evaluator_test, test_state, params)
                metrics = {k: v.mean() for k, v in metrics.items()}
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

