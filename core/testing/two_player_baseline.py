



from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from core.common import GameFrame, two_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn


class TwoPlayerBaseline(BaseTester):
    def __init__(self, num_episodes: int, baseline_evaluator: Evaluator, baseline_params: Optional[chex.ArrayTree] = None, *args, **kwargs):
        super().__init__(num_keys=num_episodes, *args, **kwargs)
        self.num_episodes = num_episodes
        self.baseline_evaluator = baseline_evaluator
        if baseline_params is None:
            baseline_params = jnp.array([])
        self.baseline_params = baseline_params
        

    def check_size_compatibilities(self, num_devices: int) -> None:
        if self.num_episodes % num_devices != 0:
            raise ValueError(f"{self.__class__.__name__}: number of episodes ({self.num_episodes}) must be divisible by number of devices ({num_devices})")

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(self,  
        max_steps: int,
        env_step_fn: EnvStepFn, 
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        keys: chex.PRNGKey,
        state: TestState, 
        params: chex.ArrayTree
    ) -> Tuple[TestState, Dict, GameFrame, chex.Array]:

        game_fn = partial(two_player_game,
            evaluator_1 = evaluator,
            evaluator_2 = self.baseline_evaluator,
            params_1 = params,
            params_2 = self.baseline_params,
            env_step_fn = env_step_fn,
            env_init_fn = env_init_fn,
            max_steps = max_steps
        )

        results, frames, p_ids = jax.vmap(game_fn)(keys)
        frames = jax.tree_map(lambda x: x[0], frames)
        p_ids = p_ids[0]
        
        avg = results[:, 0].mean()

        metrics = {
            f"{self.name}_avg_outcome": avg
        }

        return state, metrics, frames, p_ids
        