#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import cache
from typing import Hashable, Iterable
from pymdp.envs.env import Env
import numpy as np


class MultiAgentEnv(Env):
    """ 
    A base class for multiagent environments with discrete actions and observations.
    """

    
    # Static properties of the environment (game): players, actions, observations

    @property
    @cache
    def num_players(self) -> int:
        """
        Returns number of players.

        By default derived from self.player_names.
        This needs to be constant through the course of the game.
        """
        return len(self.player_names())

    @property
    def player_names(self) -> Iterable[str]:
        """
        Returns the list of player names, determining the number of players. These may just be "1", "2", ...

        This needs to be constant through the course of the game.
        """
        raise NotImplementedError

    @property
    @cache
    def num_actions(self) -> np.ndarray[int]:
        """
        Return the number of actions available to each player.

        By default, this is determined from self.action_names, though you may override it directly.
        This needs to be constant through the course of the game.
        """
        return np.array([len(a) for a in self.actions])
    
    def actions(self) -> Iterable[Iterable[Hashable]]:
        """
        Return a tuple of actions available to each player.
        
        These may be arbitrary hashable (immutable) values, e.g. strings.
        If these are just integers, then they must be just 0..n (i.e. coincide with their number).
        This needs to be constant through the course of the game.
        """
        raise NotImplementedError
    
    @property
    @cache
    def num_observations(self) -> np.ndarray[int]:
        """
        Return the number of actions available to each player at any point in the game.

        By default, this is determined from self.action_names, though you may override it directly.
        This needs to be constant through the course of the game.
        """
        return np.array([len(a) for a in self.actions])

    def observations(self) -> Iterable[Iterable[Hashable]]:
        """
        Return a tuple of observation values available to each player at any point in the game.
        

        These may be arbitrary hashable (immutable) values, e.g. strings.
        If these are just integers, then they must be just 0..n (i.e. coincide with their number).
        This needs to be constant through the course of the game.
        """
        raise NotImplementedError

    @property
    @cache
    def num_states(self) -> int:
        """
        Return the number of actions available to each player at any point in the game.

        By default, this is determined from self.state_names, though you may override it directly.
        This needs to be constant through the course of the game.
        """
        return len(self.state_names())

    def state_names(self) -> Iterable[Hashable]:
        """
        Return an iterable of possible hidden states.

        These may be arbitrary hashable (immutable) values, e.g. strings, tuples, ...
        If these are just integers, then they must be just 0..n (i.e. coincide with their number).
        This needs to be constant through the course of the game.
        """
        raise NotImplementedError

    def reset(self, rng_key=None) -> Iterable[int]:
        """
        Resets the initial state of the environment (including any initial randomization) and returns initial observation indexes.

        Accepts either an action index or a action value for each player.
        The return value may be a numpy or jax array.
        Optionally resets the internal Prng key to the given value.
        """
        raise NotImplementedError

    def step(self, actions: Iterable[Hashable]) -> Iterable[int]:
        """
        Steps the environment forward using a tuple of actions and returns the observation for each player. 

        Accepts either an action index or a action value for each player.
        The return value may be a numpy or jax array.
        """
        raise NotImplementedError
    
    def rewards(self) -> Iterable[float]:
        """
        Returns the payoff of each payer in this state.
        
        Note that in some games these should not be considered as observed.
        """
        raise NotImplementedError

    def __str__(self):
        return f"<{self.__class__.__name__}>"
    
