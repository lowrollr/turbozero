



from functools import partial
from typing import Any, Dict, Optional, Tuple
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

    def init(self, params: chex.ArrayTree, **kwargs) -> TwoPlayerTestState:
        return TwoPlayerTestState(best_params=params)
    
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
        state: TwoPlayerTestState,
        params: chex.ArrayTree
    ) -> Tuple[TwoPlayerTestState, Dict, chex.ArrayTree]:

        game_fn = partial(two_player_game,
            evaluator_1 = evaluator,
            evaluator_2 = evaluator,
            params_1 = params,
            params_2 = state.best_params,
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

        best_params = jax.lax.cond(
            avg > 0.0,
            lambda _: params,
            lambda _: state.best_params,
            None
        )

        return state.replace(best_params=best_params), metrics, frames, p_ids
