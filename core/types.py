





from typing import Callable, Tuple
import chex
import jax
from flax.training.train_state import TrainState

from core.memory.replay_memory import BaseExperience

EnvStepFn = Callable[[chex.ArrayTree, int], Tuple[chex.ArrayTree, float, bool]]
EnvInitFn = Callable[[jax.random.PRNGKey], chex.ArrayTree]  
EnvPlayerIdFn = Callable[[chex.ArrayTree], int]
ActionMaskFn = Callable[[chex.ArrayTree], chex.Array]
Params = chex.ArrayTree
EvalFn = Callable[[chex.ArrayTree, Params], Tuple[chex.Array, float]]
PartialEvaluationFn = Callable[[chex.ArrayTree], Tuple[chex.Array, float]]
TrainStepFn = Callable[[BaseExperience, TrainState], TrainState]
ExtractModelParamsFn = Callable[[TrainState], chex.ArrayTree]