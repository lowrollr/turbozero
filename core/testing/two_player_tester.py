



from functools import partial
from typing import Dict, Optional, Tuple
from chex import dataclass
import chex
import jax
from core.common import two_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn

@dataclass(frozen=True)
class TwoPlayerTestState(TestState):
    best_params: chex.ArrayTree

class TwoPlayerTester(BaseTester):
    def __init__(self, num_episodes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_episodes = num_episodes

    def init(self, key: jax.random.PRNGKey, params: chex.ArrayTree, **kwargs) -> TwoPlayerTestState:
        return TwoPlayerTestState(key=key, best_params=params)
    
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
    ) -> Tuple[TwoPlayerTestState, Dict]:
        key, subkey = jax.random.split(state.key)
        num_episodes = self.num_episodes // num_partitions
        game_keys = jax.random.split(subkey, num_episodes)

        game_fn = partial(two_player_game,
            evaluator_1 = evaluator,
            evaluator_2 = evaluator,
            params_1 = params,
            params_2 = state.best_params,
            env_step_fn = env_step_fn,
            env_init_fn = env_init_fn   
        )

        results = jax.vmap(game_fn)(game_keys)
        
        wins = (results[:, 0] > results[:, 1]).sum()
        draws = (results[:, 0] == results[:, 1]).sum()
        
        win_rate = (wins + (0.5 * draws)) / num_episodes

        metrics = {
            "performance_vs_best": win_rate
        }

        best_params = jax.lax.cond(
            win_rate > 0.5,
            lambda _: params,
            lambda _: state.best_params,
            None
        )

        return state.replace(key=key, best_params=best_params), metrics




    
