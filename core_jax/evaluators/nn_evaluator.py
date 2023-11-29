

from typing import Any
from core_jax.evaluators.evaluator import Evaluator, EvaluatorConfig
from core_jax.envs.env import Env, EnvState
from core_jax.evaluators.evaluator import EvaluatorState
from flax import struct
from flax import linen as nn
import jax
import jax.numpy as jnp

# just an evaluator that contains a neural network
# I think it makes sense for the evaluator state to own the model weights 
# rather than the training state
# but others seem to disagree
class NNEvaluator(Evaluator):
    def __init__(self, 
        env: Env, 
        config: EvaluatorConfig, 
        model: nn.Module, 
        **kwargs
    ):
        super().__init__(env, config, **kwargs)
        self.model = model
    
    
    def init_params(self, key: jax.random.PRNGKey) -> struct.PyTreeNode:
        return self.model.init(key, jnp.zeros((1, *self.env.get_observation_shape())))

    def predict(self,
        observation: struct.PyTreeNode,
        model_params: struct.PyTreeNode
    ) -> Any:
        return self.model.apply(model_params, observation[None, ...], training=False)


    def evaluate(self, 
        state: EvaluatorState,
        env_state: EnvState,
        model_params: struct.PyTreeNode,
        **kwargs    
    ) -> EvaluatorState:
        return super().evaluate(state, env_state, model_params=model_params, **kwargs)