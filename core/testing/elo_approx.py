
from functools import partial
from typing import Dict, List, Optional
import chex
import jax
import optax
import jax.numpy as jnp
from core.common import two_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn

@chex.dataclass(frozen=True)
class ApproxEloTesterState(TestState):
    past_opponent_params: chex.ArrayTree
    next_idx: int
    generation: int
    past_opponent_ids: chex.Array
    matchup_matrix: chex.Array

class ApproxEloTester(BaseTester):
    def __init__(self, 
        episodes_per_opponent: int,
        num_opponenets: int, 
        elo_base: float = 10.0,
        elo_exp_divisor: float = 400.0,
        baseline_rating: float = 0.0,
        rating_optim_lr: Optional[float] = None,
        rating_optim_steps: int = 10,
        *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episodes_per_opponent = episodes_per_opponent
        self.num_opponents = num_opponenets
        self.elo_base = elo_base
        self.elo_exp_divisor = elo_exp_divisor
        self.baseline_rating = baseline_rating 
        self.rating_optim_lr = rating_optim_lr if rating_optim_lr is not None else elo_exp_divisor        
        self.rating_optim_steps = rating_optim_steps
    
    def init(self, key: jax.random.PRNGKey, params: chex.ArrayTree) -> chex.ArrayTree:
        return ApproxEloTesterState(
            key=key, 
            past_opponent_params=jax.tree_util.tree_map(
                lambda x: jnp.stack([x] * self.num_opponents),
                params
            ),
            next_idx = 0,
            generation = 1,
            past_opponent_ids = jnp.zeros(self.num_opponents, dtype=jnp.int32),
            matchup_matrix = jnp.zeros((1,1,2), dtype=jnp.float32)
        )

    def test(self,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        evaluator: Evaluator,
        state: TestState,
        params: chex.ArrayTree         
    ) -> [ApproxEloTesterState, Dict]:
        num_episodes = self.episodes_per_opponent * self.num_opponents

        key, subkey = jax.random.split(state.key)
        game_keys = jax.random.split(subkey, num_episodes)
        
        game_fn = partial(two_player_game,
            evaluator_1 = evaluator,
            evaluator_2 = evaluator,
            params_1 = params,
            env_step_fn = env_step_fn,
            env_init_fn = env_init_fn
        )
        
        opponent_params = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, (self.episodes_per_opponent, *([1] * (len(x.shape) - 1)))),
            state.past_opponent_params
        )

        opp_ids = jnp.tile(state.past_opponent_ids, self.episodes_per_opponent)

        results = jax.vmap(game_fn)(game_keys, params_2=opponent_params)

        wins = results[:, 0] > results[:, 1]
        draws = results[:, 0] == results[:, 1]


        matchup_matrix = jnp.pad(state.matchup_matrix, ((0,1), (0,1), (0,0)))

        def add_results(idx, matchup_matrix):
            opp_id = state.past_opponent_ids[idx]
            scores = jnp.where(
                jnp.logical_and(wins, opp_id == opp_ids),
                1.0,
                jnp.where(
                    jnp.logical_and(draws, opp_id == opp_ids),
                    0.5,
                    0.0
                )
            ).sum()
            total_games = jnp.sum(opp_id == opp_ids)
            matchup_matrix = matchup_matrix.at[-1, opp_id, 0].add(scores)
            matchup_matrix = matchup_matrix.at[-1, opp_id, 1].add(total_games)
            return matchup_matrix
        

        matchup_matrix = jax.lax.fori_loop(0, self.num_opponents, add_results, matchup_matrix)
        

        elo_ratings = self.get_elo_rating(matchup_matrix)

        metrics = {
            "elo_rating": elo_ratings[-1],
        }
    
        new_opponent_params = jax.tree_util.tree_map(
            lambda x, y: x.at[state.next_idx].set(y),
            state.past_opponent_params,
            params
        )

        new_opponent_ids = state.past_opponent_ids.at[state.next_idx].set(state.generation)
    

        return state.replace(
            key=key,
            past_opponent_params = new_opponent_params,
            next_idx = (state.next_idx + 1) % self.num_opponents,
            generation = state.generation + 1,
            matchup_matrix = matchup_matrix,
            past_opponent_ids = new_opponent_ids
        ), metrics
    

    def get_elo_rating(self, matchup_matrix) -> float:
        matrix_mask = matchup_matrix[:, :, 1] > 0
        win_pct_matrix = jnp.where(
            matrix_mask,
            matchup_matrix[:, :, 0] / matchup_matrix[:, :, 1],
            jnp.zeros_like(matchup_matrix[:, :, 0], dtype=jnp.float32)
        )

        ratings = jnp.full((matchup_matrix.shape[0],), self.baseline_rating, dtype=jnp.float32)

        optimizer = optax.sgd(learning_rate=self.rating_optim_lr)

        optimizer_state = optimizer.init(ratings)

        def expected_result(rating, opp_rating):
            return 1.0 / (1.0 + jnp.power(self.elo_base, (opp_rating - rating) / self.elo_exp_divisor))

        def loss_fn(ratings):
            expected_results = expected_result(ratings.reshape(-1, 1), ratings.reshape(1,-1))
            diff = win_pct_matrix - expected_results
            return jnp.mean(jnp.square(diff) * matrix_mask)
        
        def step(ratings, opt_state, _):
            loss, grad = jax.value_and_grad(loss_fn)(ratings)
            updates, opt_state = optimizer.update(grad, opt_state, ratings)
            new_ratings = optax.apply_updates(ratings, updates)
            return new_ratings, opt_state, loss
        
        ratings, _, _ = jax.lax.fori_loop(0, self.rating_optim_steps, lambda _, x: step(*x), (ratings, optimizer_state, 0.0))

        ratings = ratings - jnp.min(ratings)

        return ratings
