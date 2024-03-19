

from functools import partial
from typing import Dict, Tuple
import chex
import jax
from core.common import single_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn


class SinglePlayerTester(BaseTester):
    def __init__(self, num_episodes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_episodes = num_episodes

    def check_size_compatibilities(self, num_devices: int) -> None:
        if self.num_episodes % num_devices != 0:
            raise ValueError(f"{self.__class__.__name__}: number of episodes ({self.num_episodes}) must be divisible by number of devices ({num_devices})")

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(self, 
        env_step_fn: EnvStepFn, 
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        num_partitions: int,
        state: TestState, 
        params: chex.ArrayTree
    ) -> Tuple[TestState, Dict]:
        key, subkey = jax.random.split(state.key)
        num_episodes = self.num_episodes // num_partitions
        game_keys = jax.random.split(subkey, num_episodes)

        game_fn = partial(single_player_game,
            evaluator = evaluator,
            params = params,
            env_step_fn = env_step_fn,
            env_init_fn = env_init_fn   
        )

        rewards = jax.vmap(game_fn)(game_keys)

        metrics = {'mean_reward': rewards.mean()}

        return state.replace(key=key), metrics
        