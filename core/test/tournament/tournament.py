


from collections import defaultdict
from dataclasses import dataclass
import logging
from random import shuffle
import random
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from core.algorithms.evaluator import Evaluator
from itertools import combinations
from core.algorithms.load import init_evaluator
from core.env import Env
from core.utils.checkpoint import load_checkpoint, load_model_and_optimizer_from_checkpoint
from core.utils.heatmap import annotate_heatmap, heatmap
from core.test.tester import collect_games

from envs.load import init_env

class TournamentPlayer:
    def __init__(self, name: str, evaluator: Evaluator, raw_config: dict, initial_rating=1500) -> None:
        self.rating: float = initial_rating
        self.initial_rating = initial_rating
        self.name = name
        self.evaluator = evaluator
        self.raw_config = raw_config
    
    def expected_result(self, opponent_rating: float):
        return 1 / (1 + 10 ** ((opponent_rating - self.rating) / 400))
    
    def update_rating(self, opponent_rating: float, result: float):
        self.rating += 16 * (result - self.expected_result(opponent_rating))

    def reset_rating(self):
        self.rating = self.initial_rating


@dataclass
class GameResult:
    player1_name: str
    player2_name: str
    player1_result: float
    player2_result: float


class Tournament:
    def __init__(self, env: Env, n_games: int, n_tournaments: int, device: torch.device, name: str, debug: bool = False):
        self.competitors = []
        self.competitors_dict = dict()
        self.env = env
        self.n_games = n_games
        self.n_tournaments = n_tournaments
        self.results: List[GameResult] = []
        self.device = device
        self.name = name
        self.debug = debug
    
    def init_competitor(self, config: dict) -> TournamentPlayer:
        if config.get('checkpoint'):
            model, _ = load_model_and_optimizer_from_checkpoint(load_checkpoint(config['checkpoint']), self.env, self.device)
            evaluator = init_evaluator(config['algo_config'], self.env, model)
        else:
            evaluator = init_evaluator(config['algo_config'], self.env)
        return TournamentPlayer(config['name'], evaluator, config)
    
    def collect_games(self, new_competitor_config: dict):
        new_competitor = self.init_competitor(new_competitor_config)
        if new_competitor.name not in self.competitors_dict:
            for competitor in self.competitors:
                competitor.evaluator.env = self.env
                new_competitor.evaluator.env = self.env
                p1_scores = collect_games(competitor.evaluator, new_competitor.evaluator, self.n_games, self.device, self.debug)
                new_results = []
                for p1_score in p1_scores:
                    new_results.append(GameResult(
                        player1_name=competitor.name,
                        player2_name=new_competitor.name,
                        player1_result=p1_score,
                        player2_result=1 - p1_score
                    ))
                logging.info(f'{competitor.name}: {sum([r.player1_result for r in new_results])}, {new_competitor.name}: {sum([r.player2_result for r in new_results])}')
                self.results.extend(new_results)
            self.competitors.append(new_competitor)
            self.competitors_dict[new_competitor.name] = new_competitor
        else:
            logging.warn(f'Already have data for competitor {new_competitor.name}, skipping...')

    def remove_competitor(self, name: str):
        self.competitors = [competitor for competitor in self.competitors if competitor.name != name]
        self.competitors_dict.pop(name)
        self.results = [result for result in self.results if result.player1_name != name and result.player2_name != name]

    def save(self, path: Optional[str] = '') -> None:
        if not path:
            path = f'{self.name}.pt'
        data = dict()
        data['competitor_configs'] = dict()
        for competitor in self.competitors:
            data['competitor_configs'][competitor.name] = competitor.raw_config
        data['env_config'] = self.env.config.__dict__
        data['n_games'] = self.n_games
        data['n_tournaments'] = self.n_tournaments
        data['results'] = self.results
        torch.save(data, path)

    def simulate_elo(self, interactive: bool = True) -> Dict[str, int]:
        player_ratings = defaultdict(lambda: [])
        matchups: Dict[Tuple[str, str], float] = defaultdict(lambda: 0)
        for _ in range(self.n_tournaments):
            shuffle(self.results)
            for result in self.results:
                self.competitors_dict[result.player1_name].update_rating(self.competitors_dict[result.player2_name].rating, result.player1_result)
                self.competitors_dict[result.player2_name].update_rating(self.competitors_dict[result.player1_name].rating, result.player2_result)
            for competitor in self.competitors:
                player_ratings[competitor.name].append(competitor.rating)
                competitor.reset_rating()
        for result in self.results:
            matchups[(result.player1_name, result.player2_name)] += result.player1_result
            matchups[(result.player2_name, result.player1_name)] += result.player2_result
        for key, value in matchups.items():
            matchups[key] = (value / self.n_games) * 100

        matchup_matrix = np.zeros((len(self.competitors), len(self.competitors)))
        
        final_ratings = {name: int(sum(ratings) / len(ratings)) for name, ratings in player_ratings.items()}

        sorted_competitors = sorted(self.competitors, key=lambda c: final_ratings[c.name])
        for p1_idx in range(len(sorted_competitors)):
            for p2_idx in range(p1_idx+1, len(sorted_competitors)):
                p1_name = sorted_competitors[p1_idx].name
                p2_name = sorted_competitors[p2_idx].name
                matchup_matrix[p1_idx, p2_idx] = matchups[(p1_name, p2_name)]
                matchup_matrix[p2_idx, p1_idx] = matchups[(p2_name, p1_name)]
        

        logging.info(f'Final ratings: {final_ratings}')
        player_names = [c.name for c in sorted_competitors]
        player_names_elo = [f'{c.name} ({final_ratings[c.name]})' for c in sorted_competitors]


        if interactive: 
            height = len(player_names) / 1.5
            width = len(player_names) * 1.5
            fig, ax = plt.subplots(figsize=(width, height), dpi=500)
            im, cbar = heatmap(matchup_matrix, player_names_elo, player_names, ax=ax,
                   cmap="YlGn", cbarlabel="Head-to-Head Win Rate (%)")
            texts = annotate_heatmap(im, valfmt="{x:.1f}%")
            fig.tight_layout()
            plt.show()

        return final_ratings
    
    def run(self, competitors: List[dict], interactive: bool = True): 
        for competitor in competitors:
            self.collect_games(competitor)
        self.save()
        return self.simulate_elo(interactive)

def load_tournament(path: str, device: torch.device):
    tournament_data = torch.load(path)
    tournament = Tournament(
        init_env(device, tournament_data['n_games'], tournament_data['env_config'], False),
        tournament_data['n_games'],
        tournament_data['n_tournaments'],
        device,
        tournament_data.get('name', 'tournament')
    )
    tournament.results = tournament_data['results']
    for competitor_config in tournament_data['competitor_configs'].values():
        tournament.competitors.append(tournament.init_competitor(competitor_config))
    tournament.competitors_dict = {competitor.name: competitor for competitor in tournament.competitors}
    
    return tournament



    
