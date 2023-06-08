import gymnasium as gym
import numpy as np
import torch
from typing import Iterable

class SpStochasticMCTSEnv(gym.Env):
    
    # given an action, returns the potential next states, along with the likelihood of reaching each state
    def get_progressions(self, action):
        raise NotImplementedError
        # returns list of ()

    
    # returns the legal actions in the current state, 
    def get_legal_actions(self):
        raise NotImplementedError
        # returns mask of legal actions


    def _get_obs(self):
        raise NotImplementedError
        # returns observation

    def _get_info(self):
        raise NotImplementedError
        # returns info
    