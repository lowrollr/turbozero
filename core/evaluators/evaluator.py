


from typing import Callable, Dict
import chex
from chex import dataclass
import jax

@dataclass(frozen=True)
class EvalOutput:
    eval_state: chex.ArrayTree
    action: int

class Evaluator:
    def init(self, key: jax.random.PRNGKey, **kwargs) -> chex.ArrayTree:
        raise NotImplementedError()

    def reset(self, state: chex.ArrayTree) -> chex.ArrayTree:
        raise NotImplementedError()
    
    def evaluate(self, eval_state: chex.ArrayTree, env_state: chex.ArrayTree, **kwargs) -> EvalOutput:
        raise NotImplementedError()

    def step(self, state: chex.ArrayTree, action: chex.Array) -> chex.ArrayTree:
        return state
    
    def get_config(self) -> Dict:
        return {}
    

def make_stateless_evaluator(evaluation_fn: Callable[[chex.ArrayTree], EvalOutput]) -> Evaluator:
    class StatelessEvaluator(Evaluator):
        def evaluate(self, eval_state: chex.ArrayTree, env_state: chex.ArrayTree, **kwargs) -> EvalOutput:
            return evaluation_fn(eval_state, env_state)
        def reset(self, state: chex.ArrayTree) -> chex.ArrayTree:
            return state
        def init(self, key: jax.random.PRNGKey, **kwargs) -> chex.ArrayTree:
            return key
    return StatelessEvaluator()
