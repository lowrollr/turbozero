
from functools import partial
from typing import Callable, Dict, Optional, Tuple
from chex import dataclass
import chex
import jax
from core.common import partition
from core.evaluators.evaluator import Evaluator

from core.types import EnvInitFn, EnvStepFn

@dataclass(frozen=True)
class TestState:
    pass

class BaseTester:
    def __init__(self, num_keys: int, epochs_per_test: int = 1, render_fn: Optional[Callable] = None, render_dir: str = '/tmp/turbozero/', name: Optional[str] = None):
        self.num_keys = num_keys
        self.epochs_per_test = epochs_per_test
        self.render_fn = render_fn
        self.render_dir = render_dir
        if name is None:
            name = self.__class__.__name__
        self.name = name

    def init(self, **kwargs) -> TestState:
        return TestState()
    
    def check_size_compatibilities(self, num_devices: int) -> None:
        pass

    def split_keys(self, key: chex.PRNGKey, num_devices: int) -> chex.PRNGKey:
        # partition keys across devices (do this here so its reproducible no matter the number of devices used)
        keys = jax.random.split(key, self.num_keys)
        keys = partition(keys, num_devices)
        return keys

    def run(self, 
        key: chex.PRNGKey, 
        epoch_num: int, 
        max_steps: int, 
        num_devices: int, 
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        state: TestState,
        params: chex.ArrayTree,
        *args
    ) -> Tuple[TestState, Dict, str]:
        
        keys = self.split_keys(key, num_devices)

        if epoch_num % self.epochs_per_test == 0:
            state, metrics, frames, p_ids = self.test(max_steps, env_step_fn, \
                                                      env_init_fn, evaluator, keys, state, params)
            # get one set of frames
            frames = jax.tree_map(lambda x: x[0], frames)
            p_ids = p_ids[0]
            if self.render_fn is not None:
                frame_list = [jax.device_get(jax.tree_map(lambda x: x[i], frames)) for i in range(max_steps)]
                path_to_rendering = self.render_fn(frame_list, p_ids, f"{self.name}_{epoch_num}", self.render_dir)
            else:
                path_to_rendering = None
            return state, metrics, path_to_rendering
        
    
    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(self, 
        max_steps: int,
        env_step_fn: EnvStepFn, 
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        keys: chex.PRNGKey,
        state: TestState, 
        params: chex.ArrayTree
    ) -> Tuple[TestState, Dict, chex.ArrayTree]:
        raise NotImplementedError()

