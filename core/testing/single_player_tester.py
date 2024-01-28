

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

    def test(self, 
        env_step_fn: EnvStepFn, 
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        state: TestState, 
        params: chex.ArrayTree
    ) -> Tuple[TestState, Dict]:
        key, subkey = jax.random.split(state.key)
        game_keys = jax.random.split(subkey, self.num_episodes)

        game_fn = partial(single_player_game,
            evaluator = evaluator,
            params = params,
            env_step_fn = env_step_fn,
            env_init_fn = env_init_fn   
        )

        rewards = jax.vmap(game_fn)(game_keys)

        metrics = {'mean_reward': rewards.mean()}

        return state.replace(key=key), metrics
        