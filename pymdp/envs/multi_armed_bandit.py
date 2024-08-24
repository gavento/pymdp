#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Multi-Armed bandit environment

__author__: Tomáš Gavenčiak

"""

from pymdp.envs import Env
from pymdp import utils, maths
import numpy as np

class MultiArmedBanditEnv(Env):
    def __init__(self, n: int):
        self.n = n
        self.INITIAL_STATE = self.n # Initial state is the last one
        self._state = utils.obj_array_from_list([self.INITIAL_STATE])
        
        self.num_states = [self.n + 1] # locations
        self.num_locations = self.num_states[0]
        self.num_controls = self.n # Only non-inital locations
        self.num_obs = [self.n + 1, self.n + 1] # locations, rewards
        self.num_factors = len(self.num_states) # One factor for locations
        self.num_modalities = len(self.num_obs) # Two modalities: locations, rewards

        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()

    def reset(self):
        self._state = self._construct_state([self.INITIAL_STATE])
        return self._get_observation()

    def _get_observation(self):
        # Taken from TMaze
        prob_obs = [maths.spm_dot(A_m, self._state) for A_m in self._likelihood_dist]
        obs = [utils.sample(po_i) for po_i in prob_obs]
        return obs
    
    def _construct_state(self, state_tuple):
        # Taken from TMaze
        state = utils.obj_array(self.num_factors)
        for f, ns in enumerate(self.num_states):
            state[f] = utils.onehot(state_tuple[f], ns)
        return state
   
    def step(self, actions):
        # Taken from TMaze
        prob_states = utils.obj_array(self.num_factors)
        for factor, state in enumerate(self._state):
            prob_states[factor] = self._transition_dist[factor][:, :, int(actions[factor])].dot(state)
        state = [utils.sample(ps_i) for ps_i in prob_states]
        self._state = self._construct_state(state)
        return self._get_observation()

    def get_likelihood_dist(self):
        return self._likelihood_dist

    def get_transition_dist(self):
        return self._transition_dist

    def _construct_transition_dist(self):
        B = utils.obj_array_zeros([self.num_states + self.num_states + [self.num_controls]])
        for i in range(self.n):
            B[0][i,:,i] = 1.0
        return B

    def _construct_likelihood_dist(self):
        A = utils.obj_array_zeros([[self.num_obs[0]] + self.num_states, [self.num_obs[1]] + self.num_states])
        for i in range(self.n + 1):
            A[0][i,i] = 1.0
            A[1][i,i] = 1.0
        return A        

