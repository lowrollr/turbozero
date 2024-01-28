
from functools import partial
from typing import Any, Optional, Tuple
import chex
from chex import dataclass
import jax
import jax.numpy as jnp
import wandb
from core.testing.common import two_player_game

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
    def test(self, 
        key: jax.random.PRNGKey,
        params: chex.ArrayTree,
        num_episodes: int,    
        best_params: chex.ArrayTree
    ) -> Tuple[CollectionState, chex.ArrayTree, dict]:
        
        keys = jax.random.split(key, num_episodes)
        game_fn = partial(two_player_game,
            evaluator_1 = self.evaluator_test,
            evaluator_2 = self.evaluator_test,
            params_1 = params,
            params_2 = best_params,
            env_step_fn = self.env_step_fn,
            env_init_fn = self.env_init_fn   
        )

        results = jax.vmap(game_fn)(keys)

        wins = (results[:, 0] > results[:, 1]).sum()
        draws = (results[:, 0] == results[:, 1]).sum()
        
        win_rate = wins + (0.5 * draws) / num_episodes

        metrics = {
            "performance_vs_best": win_rate
        }

        best_params = jax.lax.cond(
            win_rate > 0.5,
            lambda _: params,
            lambda _: best_params,
            None
        )

        return metrics, best_params
    
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
        collection_state: Optional[CollectionState] = None,
        wandb_run: Optional[Any] = None,
        extra_wandb_config: Optional[dict] = {}
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
            collection_state=collection_state,
            wandb_run=wandb_run,
            extra_wandb_config=extra_wandb_config
        )
    