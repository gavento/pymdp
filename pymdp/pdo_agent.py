#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

from functools import partial
import itertools
import warnings
import jax
import numpy as np
from pymdp import inference, control, learning
from pymdp import utils, maths
import copy
import logging
import jax.numpy as jnp
from pymdp.agent import Agent
from pymdp.envs.env import Env
from .pdo_policy import TabularPolicy, Policy, UniformPDOPolicy

class PDOAgent(Agent):
    """ 
    The agent only runs inference on the first time it is called, then just follows the policy it inferred.
    """

    def __init__(self, A, B, time_horizon, env: Env, policy_iterations=1000, policy_lr=0.01, **kwargs):
        super(PDOAgent, self).__init__(A, B, **kwargs)
        self.policy = None
        self.action = None
        self.time_horizon = time_horizon
        self.policy_iterations = policy_iterations
        self.policy_lr = policy_lr
        self.prior_policy = UniformPDOPolicy(action_counts=self.num_controls)
        self.env = env

    def infer_policies(self):
        if self.policy is not None:
            return  # Policy has already been inferred for this agent.
        assert self.curr_timestep == 0

        # Generate all possible observartion sequences. Each onbservation is a tuple of numbers
        # Each of those numbers ranges in range(self.num_obs[i])
        possible_observations = list(itertools.product(*[range(onum) for onum in self.num_obs]))
        assert len(possible_observations) == np.prod(self.num_obs)
        all_observation_seqs = tuple(itertools.chain(*[
            itertools.product(possible_observations, repeat=rep, )
            for rep in range(self.time_horizon + 1)]))
        self.policy = TabularPolicy(action_counts=self.num_controls, observation_sequences=all_observation_seqs)

        @jax.jit
        def step_fn(policy):
            gradG = jax.grad(self.G)(policy)
            print(self.policy_lr, gradG, gradG.table)
            pol_table = policy.table - self.policy_lr * gradG.table
            return policy.updated_copy(pol_table)
        
        logging.info(f"Running policy iteration:\n{0:03d}/{self.policy_iterations:03d}: G = {self.G(self.policy):.3f}")
        for i in range(self.policy_iterations):
            self.policy = step_fn(self.policy)
            logging.info(f"{i+1:03d}/{self.policy_iterations:03d}: G = {self.G(self.policy):.3f}")

    @partial(jax.jit, static_argnums=(0,))
    def G(self, policy: Policy):
        "This needs to be JAX-differentiable wrt values from `policy`"
        return 0.0

    def infer_states(self, observation, distr_obs=False):
        assert distr_obs is False, "This agent does not support distributed observations."
        # print(self.prev_obs, self.curr_timestep)
        assert self.curr_timestep == len(self.prev_obs)
        self.prev_obs.append(tuple(int(x) for x in observation))

    def sample_action(self):
        assert self.policy is not None, "No policy has been inferred for this agent."
        self.action = self.policy.sample_action_for_observations(tuple(self.prev_obs))
        self.step_time()
        return self.action
