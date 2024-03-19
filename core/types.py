
from typing import Callable, Tuple
import chex
import jax
from flax.training.train_state import TrainState
import optax

from core.memory.replay_memory import BaseExperience

@chex.dataclass(frozen=True)
class StepMetadata:
    rewards: chex.Array
    action_mask: chex.Array
    terminated: bool
    cur_player_id: int
    

EnvStepFn = Callable[[chex.ArrayTree, int], Tuple[chex.ArrayTree, StepMetadata]]
EnvInitFn = Callable[[jax.random.PRNGKey], Tuple[chex.ArrayTree, StepMetadata]]  
Params = chex.ArrayTree
EvalFn = Callable[[chex.ArrayTree, Params, jax.random.PRNGKey], Tuple[chex.Array, float]]
LossFn = Callable[[chex.ArrayTree, TrainState, BaseExperience], Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]]
ExtractModelParamsFn = Callable[[TrainState], chex.ArrayTree]
StateToNNInputFn = Callable[[chex.ArrayTree], chex.Array]
