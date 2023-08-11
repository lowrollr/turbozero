


from collections import defaultdict
from dataclasses import dataclass
import logging
from random import shuffle
from typing import Dict, List

import torch
from core.algorithms.evaluator import Evaluator
from itertools import combinations
from core.env import Env

from envs.load import init_env

class TournamentPlayer:
    def __init__(self, name: str, evaluator: Evaluator, initial_rating=1500) -> None:
        self.rating: float = initial_rating
        self.initial_rating = initial_rating
        self.name = name
        self.evaluator = evaluator
    
    def expected_result(self, opponent_rating: float):
        return 1 / (1 + 10 ** ((opponent_rating - self.rating) / 400))
    
    def update_rating(self, opponent_rating: float, result: float):
        self.rating += 16 * (result - self.expected_result(opponent_rating))

    def reset_rating(self):
        self.rating = self.initial_rating


@dataclass
class UnratedGameResult:
    player1_name: str
    player2_name: str
    player1_result: float
    player2_result: float


class Tournament:
    def __init__(self, competitors: List[TournamentPlayer], env: Env, n_games: int, n_tournaments: int):
        self.competitors = competitors
        self.competitors_dict = {player.name: player for player in competitors}
        self.env = env
        self.n_games = n_games
        self.n_tournaments = n_tournaments

    def play_games(self, evaluator1, evaluator2) -> List[float]: # assumes zero-sum, 2 player game
        evaluator1.env = self.env
        evaluator2.env = self.env
        evaluator1.reset()
        evaluator2.reset()
        split = self.n_games // 2
        reset = torch.zeros(self.n_games, dtype=torch.bool, device=self.env.device, requires_grad=False)
        reset[:split] = True
        completed_episodes = torch.zeros(self.n_games, dtype=torch.bool, device=self.env.device, requires_grad=False)
        scores = torch.zeros(self.n_games, dtype=torch.float32, device=self.env.device, requires_grad=False)
        _, _, _, actions, terminated = evaluator1.step()
        envs_to_reset = terminated | reset
        evaluator2.step_evaluator(actions, envs_to_reset)
        evaluator1.env.terminated[:split] = True
        evaluator1.env.reset_terminated_states()
        starting_players = (evaluator1.env.cur_players.clone() - 1) % 2
        use_second_evaluator = True
        while not completed_episodes.all():
            if use_second_evaluator:
                _, _, _, actions, terminated = evaluator2.step()
                evaluator1.step_evaluator(actions, terminated)
            else:
                _, _, _, actions, terminated = evaluator1.step()
                evaluator2.step_evaluator(actions, terminated)
            rewards = evaluator1.env.get_rewards(starting_players)
            scores += rewards * terminated * (~completed_episodes)
            completed_episodes |= terminated
            use_second_evaluator = not use_second_evaluator
        return scores.cpu().tolist()

    def gather_round_robin_games(self, games_against_each_player: int) -> List[UnratedGameResult]:
        results = []
        matchups = list(combinations(self.competitors, 2))
        shuffle(matchups)
        for players in matchups:
            players = list(players)
            shuffle(players)
            player1, player2 = players
            p1_scores = self.play_games(player1.evaluator, player2.evaluator)
            new_results = []
            for p1_score in p1_scores:
                new_results.append(UnratedGameResult(
                    player1_name=player1.name,
                    player2_name=player2.name,
                    player1_result=p1_score,
                    player2_result=1 - p1_score
                ))
            results.extend(new_results)
            logging.info(f'{player1.name}: {sum([r.player1_result for r in new_results])}, {player2.name}: {sum([r.player2_result for r in new_results])}')
        return results

    def run(self) -> Dict[str, int]:
        player_ratings = defaultdict(lambda: [])
        results = self.gather_round_robin_games(self.n_games)
        for _ in range(self.n_tournaments):
            shuffle(results)
            for result in results:
                self.competitors_dict[result.player1_name].update_rating(self.competitors_dict[result.player2_name].rating, result.player1_result)
                self.competitors_dict[result.player2_name].update_rating(self.competitors_dict[result.player1_name].rating, result.player2_result)
            for competitor in self.competitors:
                player_ratings[competitor.name].append(competitor.rating)
                competitor.reset_rating()
                

        final_ratings = {name: int(sum(ratings) / len(ratings)) for name, ratings in player_ratings.items()}
        logging.info(f'Final ratings: {final_ratings}')
        return final_ratings
    

        