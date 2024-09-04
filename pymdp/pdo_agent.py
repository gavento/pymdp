#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

from functools import partial
import functools
import itertools
from typing import Any, Iterable
import jax
import numpy as np
import tqdm
import logging
import jax.numpy as jnp
from pymdp.agent import Agent
from pymdp.envs.env import Env
from .pdo_policy import PDOPolicyBase, TabularPolicy, UniformPDOPolicy, TabularSoftmaxPolicy


def sum_dicts(dicts: Iterable[dict], w: Iterable[float]|None=None, *, no_w_keys=()) -> dict:
    """Sum a list of dictionaries, possibly with weights, optionally not weighting some keys"""
    keys = dicts[0].keys()
    assert all(d.keys() == keys for d in dicts)
    if w is None:
        return {k: sum(d[k] for d in dicts) for k in keys}
    else:
        assert len(w) == len(dicts)
        return {k: sum(wi * d[k] for wi, d in zip(w, dicts)) if k not in no_w_keys else sum(d[k] for d in dicts) for k in keys}


def outer_product(*arrays):
    if isinstance(arrays[0], jax.Array):
        return functools.reduce(lambda x,y: jnp.tensordot(x, y, axes=([], [])), arrays)
    elif isinstance(arrays[0], np.ndarray):
        return functools.reduce(lambda x,y: np.tensordot(x, y, axes=([], [])), arrays)
    else:
        raise ValueError(f"Arrays must be either jnp.ndarray or np.ndarray, got {type(arrays[0])}")


class BranchingAgent(Agent):
    """ 
    The agent only runs inference on the first time it is called, then just follows the policy it inferred.
    """

    def __init__(self, A, B, time_horizon, env: Env, policy_iterations: int, policy_lr=0.01,
                 prior_policy: PDOPolicyBase=None, beta=1.0, progress=True, **kwargs):
        super(BranchingAgent, self).__init__(A, B, **kwargs)
        self.policy = None
        self.action = None
        self.time_horizon = time_horizon
        self.policy_iterations = policy_iterations
        self.policy_lr = policy_lr
        self.prior_policy = UniformPDOPolicy(action_counts=self.num_controls)
        self.env = env
        self.progress = progress
        self.prior_policy = prior_policy
        self.beta = beta

        # Possible observations for each turn:
        self.possible_observations = tuple(itertools.product(*[range(onum) for onum in self.num_obs]))
        # All possible observation sequences:
        self.possible_observation_seqs = None

        # Possible actions for each turn:
        self.possible_actions = tuple(itertools.product(*[range(anum) for anum in self.num_controls]))

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
        raise NotImplementedError("This method *may* be implemented by subclasses.")

    def infer_policies(self):
        if self.policy is not None:
            return  # Policy has already been inferred for this agent.
        assert self.curr_timestep == 0

        # Generate all possible observartion sequences. Each onbservation is a tuple of numbers
        # Each of those numbers ranges in range(self.num_obs[i])
        if self.possible_observation_seqs is None:
            self.possible_observation_seqs = self.generate_consistent_observation_seqs()
            # Alternatively, use all sequences (wasteful)
            #self.possible_observation_seqs = self.generate_all_observation_seqs()

        try:
            self.policy = self.optimal_policy()
            return
        except NotImplementedError:
            pass

        self.policy = TabularSoftmaxPolicy(action_counts=self.num_controls, observation_sequences=self.possible_observation_seqs)
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
    def G(self, policy: PDOPolicyBase):
        "This needs to be JAX-differentiable wrt values from `policy.table`"
        raise NotImplementedError("This method must be implemented by subclasses.")
    
    def infer_states(self, observation, distr_obs=False):
        assert distr_obs is False, "This agent does not support distributed observations."
        assert self.curr_timestep == len(self.prev_obs)
        self.prev_obs.append(tuple(int(x) for x in observation))

    def sample_action(self):
        assert self.policy is not None, "No policy has been inferred for this agent."
        self.action = self.policy.sample_action_for_observations(tuple(self.prev_obs))
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
                evidence = np.prod([Af[o[i]] for i, Af in enumerate(self.A)], axis=0)
                sbaa2 = np.array(state_belief_all_actions) * evidence
                if np.sum(sbaa2) < eps:
                    continue
                sbaa2 = sbaa2 / np.sum(sbaa2) # Bayesian belief update - condition on observations for this method
                # Apply uniform actions to B[i]
                B3 = [np.tensordot(self.B[i], np.ones(self.num_controls[i]) / self.num_controls[i], axes=1) for i in range(self.num_factors)]
                # COnsolidate updates to state
                B3agg = outer_product(*B3).transpose(tuple(range(0, 2*self.num_factors, 2)) + tuple(range(1, 2*self.num_factors, 2)))
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


class PDOAgent(BranchingAgent):
    """Agent minimizing PDO - Work in progress"""

    def G(self, policy: PDOPolicyBase):
        "This needs to be JAX-differentiable wrt values from `policy.table`"
        B_agg = self.compute_B_agg(with_jax=True)
        uniform_policy = policy.uniform_policy()

        def _helper(observations: tuple[tuple[int]], state_prob: jnp.ndarray) -> jnp.ndarray:
            if len(observations) > self.time_horizon:
                return 0.0

            FE = jnp.array(0.0)
            for o in self.possible_observations:
                 # This couan be just an numpy product - does not depend on policy
                evidence = jnp.prod([Af[o[i]] for i, Af in enumerate(self.A)], axis=0)

                sb2 = jnp.array(state_prob) * evidence
                for i in range(self.num_modalities):
                    FE -= self.C[i][o[i]] * jnp.sum(sb2)

                obs2 = observations + (o,)
                if obs2 not in policy.observation_seq_index:
                    continue
                po2 = policy.policy_for_observations(obs2)
            
                # Prior policy
                if self.prior_policy is None or obs2 not in self.prior_policy.observation_seq_index:
                    ppo2 = uniform_policy
                else:
                    ppo2 = self.prior_policy.policy_for_observations(obs2)
                ppo2 = jnp.array(ppo2)

                # Add penalty for KL divergence between po2 and ppo2
                FE += (1 / self.beta) * jnp.sum(sb2) * jnp.sum(po2 * (jnp.log2(po2 + 1e-10) - jnp.log2(ppo2 + 1e-10)))

                # Apply policy actions to B_agg
                B_agg3 = jnp.tensordot(B_agg, po2, axes=self.num_factors)
                # Apply previous state belief to B3
                sb4 = jnp.tensordot(B_agg3, sb2, axes=self.num_factors)
                FE += _helper(obs2, sb4)

            return FE

        return _helper((), outer_product(*self.D))


class EVAgent(BranchingAgent):
    """Agent minimizing expected value of the reward (sum over all the timesteps)"""

    def G(self, policy: PDOPolicyBase):
        B_agg = self.compute_B_agg(with_jax=True)

        def _helper(observations: tuple[tuple[int]], state_prob: jnp.ndarray) -> jnp.ndarray:
            if len(observations) > self.time_horizon:
                return 0.0

            FE = jnp.array(0.0)
            for o in self.possible_observations:
                 # This can be just NP product - does not depend on policy
                evidence = np.prod([Af[o[i]] for i, Af in enumerate(self.A)], axis=0)

                sb2 = jnp.array(state_prob) * evidence
                for i in range(self.num_modalities):
                    FE -= self.C[i][o[i]] * jnp.sum(sb2)

                obs2 = observations + (o,)
                if obs2 not in policy.observation_seq_index:
                    continue
                po2 = policy.policy_for_observations(obs2)
            
                # Apply policy actions to B_agg
                B_agg3 = jnp.tensordot(B_agg, po2, axes=self.num_factors)
                # Apply previous state belief to B3
                sb4 = jnp.tensordot(B_agg3, sb2, axes=self.num_factors)
                FE += _helper(obs2, sb4)

            return FE

        return _helper((), outer_product(*self.D))


Observation = tuple[int, ...]
Observations = tuple[Observation, ...]
Action = tuple[int, ...]
Actions = tuple[Action, ...]
Policy = dict[Observations, np.ndarray]
Stats = dict[str, int | float | np.ndarray]


class EVAgentDirect(BranchingAgent):
    """Agent minimizing expected value of the reward (sum over all the timesteps)
    
    Note that this assumes that the actions the agent took are implied by the observations!"""

    def optimal_policy(self) -> tuple[Policy, Stats]:
        policy, stats = self._optimal_policy_rec((), (), outer_product(*self.D), {})
        self.stats = stats
        return TabularPolicy.from_dict(policy)

    def _optimal_policy_rec(self, observations: Observations, actions: Actions,
             state_probs: jnp.ndarray, _seen_observations: dict[Observations, Actions]|None=None,
             _eps:float=1e-10) -> tuple[Policy, Stats]:
        """Given a sequence of (observation, action, observation, action, ..., observation), and the hidden state probabilities,
        returns the optimal policy for this subtree, and the statistics (in particular EV and G)."""
        # print(f"{'  ' * len(observations)}Exploring observations {observations} and actions {actions}")

        assert len(observations) == len(actions)
        assert len(observations) <= self.time_horizon

        # Normalize state_probs (we already conditioned on the observations)
        state_probs = jnp.array(state_probs) / jnp.sum(state_probs)
        assert state_probs.shape == tuple(self.num_states), f"State probabilities shape {state_probs.shape} does not match expected {self.num_states}"

        # Overall policy and the policy and stats of each action
        stats = {"EV": 0.0, "G": 0.0, "nodes": 0}
        policy = {}

        # Compute the probabilities of *all* next observations in the current state
        op2s = [np.tensordot(self.A[i], state_probs, axes=self.num_factors) for i in range(self.num_modalities)]
        for i, op2 in enumerate(op2s):
            assert op2.shape == (self.num_obs[i],), f"Shape of observation[{i}] probabilities {op2.shape} does not match expected {self.num_obs[i]}"

        # Iterate over the possible observations
        for o in self.possible_observations:
            oprob = np.prod([op2s[i][o[i]] for i in range(self.num_modalities)]) 
            # Check that the likelihood of this observation is larger than epsilon
            if oprob < _eps:
                # print(f"{'  ' * len(observations)}  - Skipping observation {o} after observations {observations} and actions {actions} with probability {oprob}")
                continue
            # print(f"{'  ' * len(observations)}  + Exploring observation {o} after observations {observations} and actions {actions} with probability {oprob}")

            observations2 = observations + (o,)
            if _seen_observations is not None:
                if observations2 in _seen_observations:
                    raise ValueError(f"Observation {observations2} already reached by actions {_seen_observations[observations2]}, reached again by {actions}.")
                _seen_observations[observations2] = actions

            # Update the state_probs conditioning on the observation
            evidence = np.prod([Af[o[i]] for i, Af in enumerate(self.A)], axis=0)
            state_probs2 = state_probs * evidence
            # Normalise state_probs2, since we are conditioning on the observation
            state_probs2 = state_probs2 / np.sum(state_probs2)

            # Account for the reward of that observation
            for i in range(self.num_modalities):
                dG = -self.C[i][o[i]] * oprob
                stats["G"] += dG
                stats["EV"] -= dG
            stats["nodes"] += 1

            if len(observations2) > self.time_horizon:
                continue

            a_stats = {}
            a_policies = {}
            # iterate over the next action
            for a in self.possible_actions:
                # Select part of B_agg matching the selected action (indexing on the _last_ dimensions of B_agg)
                B_agg2 = self.B_agg[..., *a]
                # Compute the probabilities of the next state
                state_probs3 = np.tensordot(B_agg2, state_probs2, axes=self.num_factors)
                assert np.isclose(np.sum(state_probs3), 1.0, atol=1e-6), "Should still be a valid distribution"
                
                actions2 = actions + (a,)
                rec_policy, rec_stats = self._optimal_policy_rec(observations2, actions2, state_probs3, _seen_observations=_seen_observations, _eps=_eps)
                # Collect the policy and update the stats
                a_policies[a] = rec_policy
                a_stats[a] = rec_stats

            # Select the action and therefore also policy for this observation
            best_a = min(a_stats, key=lambda a: a_stats[a]["G"])
            policy.update(a_policies[best_a])
            new_pol = np.zeros(self.num_controls)
            new_pol[*best_a] = 1.0
            policy[observations2] = new_pol
            stats = sum_dicts([stats, a_stats[best_a]], w=[1.0, oprob], no_w_keys=["nodes"])

        # # Leaf observation had no actions
        # if len(observations2) > self.time_horizon:
        #     print(f"{'  ' * len(observations)}.. returning {stats}")
        #     return {}, stats
        
        # Select the best action
        # print(a_stats)
        #policy = a_policies[best_a]
        #print(f"{'  ' * len(observations)}.. selected {best_a}: {np.array_str(new_pol, max_line_width=100).replace('\n', ' ')}, returning {stats}")
        return policy, stats

