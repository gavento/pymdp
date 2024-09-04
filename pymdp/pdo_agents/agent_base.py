#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A "sophisticated" base agent for EV and PDO agents

__author__: Tomáš Gavenčiak

"""

from functools import partial
import itertools
import jax
import numpy as np
import tqdm
import jax.numpy as jnp
from pymdp.agent import Agent
from pymdp.envs.env import Env
from .common import outer_product
from .full_policy import FullPolicyBase, UniformPolicy, TabularSoftmaxPolicy


class BranchingAgent(Agent):
    """ 
    The agent only runs inference on the first time it is called, then just follows the policy it inferred.
    """

    def __init__(self, A, B, time_horizon, env: Env, policy_iterations: int, policy_lr=0.01,
                 prior_policy: FullPolicyBase = None, beta=1.0, progress=True, **kwargs):
        super(BranchingAgent, self).__init__(A, B, **kwargs)
        self.policy = None
        self.action = None
        self.time_horizon = time_horizon
        self.policy_iterations = policy_iterations
        self.policy_lr = policy_lr
        self.prior_policy = UniformPolicy(action_counts=self.num_controls)
        self.env = env
        self.progress = progress
        self.prior_policy = prior_policy
        self.beta = beta

        # Possible observations for each turn:
        self.possible_observations = tuple(
            itertools.product(*[range(onum) for onum in self.num_obs]))
        # All possible observation sequences:
        self.possible_observation_seqs = None

        # Possible actions for each turn:
        self.possible_actions = tuple(itertools.product(
            *[range(anum) for anum in self.num_controls]))

        # A has shape: (tgt_observation_dim[i] + src_state_dims) for i in range(num_modalities)
        # B has shape: (tgt_state_dim[i] + src_state_dims + action_dim[i]) for i in range(num_factors)

        self.B_agg = self.compute_B_agg()
        # The shape of B_agg is: (tgt_state_dims + src_state_dims + action_dims)

    def compute_B_agg(self, with_jax=False):
        """The shape of B_agg is: (tgt_state_dims + src_state_dims + action_dims)"""
        B = [jnp.array(b) for b in self.B] if with_jax else self.B
        B_agg = outer_product(*B)
        B_agg_d = len(B_agg.shape)
        return B_agg.transpose(tuple(range(0, B_agg_d, 3)) + tuple(range(1, B_agg_d, 3)) + tuple(range(2, B_agg_d, 3)))

    def optimal_policy(self):
        raise NotImplementedError(
            "This method *may* be implemented by subclasses.")

    def infer_policies(self):
        if self.policy is not None:
            return  # Policy has already been inferred for this agent.
        assert self.curr_timestep == 0

        # Generate all possible observartion sequences. Each onbservation is a tuple of numbers
        # Each of those numbers ranges in range(self.num_obs[i])
        if self.possible_observation_seqs is None:
            self.possible_observation_seqs = self.generate_consistent_observation_seqs()
            # Alternatively, use all sequences (wasteful)
            # self.possible_observation_seqs = self.generate_all_observation_seqs()

        try:
            self.policy = self.optimal_policy()
            return
        except NotImplementedError:
            pass

        self.policy = TabularSoftmaxPolicy(
            action_counts=self.num_controls, observation_sequences=self.possible_observation_seqs)
        self.iterate_policy(iterations=self.policy_iterations)

    def iterate_policy(self, iterations):
        @jax.jit
        def step_fn(policy):
            gradG = jax.grad(self.G)(policy)
            pol_table = policy.table - self.policy_lr * gradG.table
            return policy.updated_copy(pol_table)

        for i in (bar := tqdm.trange(iterations, disable=not self.progress, leave=True)):
            self.policy = step_fn(self.policy)
            if i % 100 == 0:
                bar.set_postfix(G=self.G(self.policy))

    @partial(jax.jit, static_argnums=(0,))
    def G(self, policy: FullPolicyBase):
        "This needs to be JAX-differentiable wrt values from `policy.table`"
        raise NotImplementedError(
            "This method must be implemented by subclasses.")

    def infer_states(self, observation, distr_obs=False):
        assert distr_obs is False, "This agent does not support distributed observations."
        assert self.curr_timestep == len(self.prev_obs)
        self.prev_obs.append(tuple(int(x) for x in observation))

    def sample_action(self):
        assert self.policy is not None, "No policy has been inferred for this agent."
        self.action = self.policy.sample_action_for_observations(
            tuple(self.prev_obs))
        self.step_time()
        return self.action

    def generate_consistent_observation_seqs(self, eps=1e-10, max_depth=None):
        if max_depth is None:
            max_depth = self.time_horizon
        reachable_observations = set()

        def _helper(observations: tuple[tuple[int]], state_belief_all_actions: np.array):
            if len(observations) >= max_depth:
                return

            for o in self.possible_observations:
                o2 = observations + (o,)
                evidence = np.prod([Af[o[i]]
                                   for i, Af in enumerate(self.A)], axis=0)
                sbaa2 = np.array(state_belief_all_actions) * evidence
                if np.sum(sbaa2) < eps:
                    continue
                # Bayesian belief update - condition on observations for this method
                sbaa2 = sbaa2 / np.sum(sbaa2)
                # Apply uniform actions to B[i]
                B3 = [np.tensordot(self.B[i], np.ones(
                    self.num_controls[i]) / self.num_controls[i], axes=1) for i in range(self.num_factors)]
                # COnsolidate updates to state
                B3agg = outer_product(
                    *B3).transpose(tuple(range(0, 2*self.num_factors, 2)) + tuple(range(1, 2*self.num_factors, 2)))
                # Apply previous state belief to B3
                sbaa4 = np.tensordot(B3agg, sbaa2, axes=self.num_factors)
                reachable_observations.add(o2)
                _helper(o2, sbaa4)

        _helper((), outer_product(*self.D))
        return tuple(sorted(reachable_observations))

    def generate_all_observation_seqs(self):
        return tuple(itertools.chain(*[
            itertools.product(self.possible_observations, repeat=rep, )
            for rep in range(1, self.time_horizon + 1)]))

    def reset(self, reset_policy=False):
        self.curr_timestep = 0
        self.prev_obs = []
        self.action = None
        if reset_policy:
            self.policy = None
