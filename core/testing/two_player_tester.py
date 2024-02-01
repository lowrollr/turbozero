



from functools import partial
from typing import Dict, Optional
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
    
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def test(self,  
        env_step_fn: EnvStepFn, 
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        state: TestState, 
        params: chex.ArrayTree
    ) -> [TwoPlayerTestState, Dict]:
        key, subkey = jax.random.split(state.key)
        game_keys = jax.random.split(subkey, self.num_episodes)

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
        
        win_rate = (wins + (0.5 * draws)) / self.num_episodes

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




    
