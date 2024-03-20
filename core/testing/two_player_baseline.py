



from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from core.common import two_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn


class TwoPlayerBaseline(BaseTester):
    def __init__(self, num_episodes: int, baseline_evaluator: Evaluator, baseline_params: Optional[chex.ArrayTree] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_episodes = num_episodes
        self.baseline_evaluator = baseline_evaluator
        if baseline_params is None:
            baseline_params = jnp.array([])
        self.baseline_params = baseline_params

    def check_size_compatibilities(self, num_devices: int) -> None:
        if self.num_episodes % num_devices != 0:
            raise ValueError(f"{self.__class__.__name__}: number of episodes ({self.num_episodes}) must be divisible by number of devices ({num_devices})")

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4, 5))
    def test(self,  
        env_step_fn: EnvStepFn, 
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        num_partitions: int,
        max_steps: int,
        state: TestState, 
        params: chex.ArrayTree
    ) -> Tuple[TestState, Dict, chex.ArrayTree]:
        key, subkey = jax.random.split(state.key)
        num_episodes = self.num_episodes // num_partitions
        game_keys = jax.random.split(subkey, num_episodes)

        game_fn = partial(two_player_game,
            evaluator_1 = evaluator,
            evaluator_2 = self.baseline_evaluator,
            params_1 = params,
            params_2 = self.baseline_params,
            env_step_fn = env_step_fn,
            env_init_fn = env_init_fn,
            max_steps = max_steps
        )

        results, frames = jax.vmap(game_fn)(game_keys)
        frames = jax.tree_map(lambda x: x[0], frames)
        
        wins = (results[:, 0] > results[:, 1]).sum()
        draws = (results[:, 0] == results[:, 1]).sum()
        
        win_rate = (wins + (0.5 * draws)) / num_episodes

        metrics = {
            "performance_vs_baseline": win_rate
        }

        return state.replace(key=key), metrics, frames
        