
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple
from chex import dataclass
import chex
import jax
from core.evaluators.evaluator import Evaluator

from core.types import EnvInitFn, EnvStepFn

@dataclass(frozen=True)
class TestState:
    key: jax.random.PRNGKey

class BaseTester:
    def __init__(self, epochs_per_test: int = 1, render_fn: Optional[Callable] = None, render_dir: str = '/tmp/turbozero/'):
        self.epochs_per_test = epochs_per_test
        self.render_fn = render_fn
        self.render_dir = render_dir

    def init(self, key: jax.random.PRNGKey, **kwargs) -> TestState:
        return TestState(key=key)
    
    def check_size_compatibilities(self, num_devices: int) -> None:
        raise NotImplementedError()

    def run(self, epoch_num: int, max_steps: int, *args) -> Tuple[TestState, Dict, str]:
        if epoch_num % self.epochs_per_test == 0:
            state, metrics, frames = self.test(max_steps, *args)
            # get one set of frames
            frames = jax.tree_map(lambda x: x[0], frames)
            if self.render_fn is not None:
                frame_list = [jax.device_get(jax.tree_map(lambda x: x[i], frames)) for i in range(max_steps)]
                path_to_rendering = self.render_fn(frame_list, f"{self.__class__.__name__}_{epoch_num}", self.render_dir)
            else:
                path_to_rendering = None
            return state, metrics, path_to_rendering
        
    
    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4, 5))
    def test(self, 
        max_steps: int,
        env_step_fn: EnvStepFn, 
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        num_partitions: int,
        state: TestState, 
        params: chex.ArrayTree
    ) -> Tuple[TestState, Dict, chex.ArrayTree]:
        raise NotImplementedError()

