#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" EV and PDO agents - G-minimization via direct computation

__author__: Tomáš Gavenčiak

"""

import numpy as np
import jax.numpy as jnp

from .common import ActionSequence, ObservationSequence, PolicyStats, PolicyDict, outer_product, sum_dicts
from .agent_base import BranchingAgent
from .full_policy import TabularPolicy


class EVAgentDirect(BranchingAgent):
    """Agent minimizing expected value of the reward (sum over all the timesteps)

    Note that this assumes that the actions the agent took are implied by the observations!"""

    def optimal_policy(self) -> tuple[PolicyDict, PolicyStats]:
        policy, stats = self._optimal_policy_rec(
            (), (), outer_product(*self.D), {})
        self.stats = stats
        return TabularPolicy.from_dict(policy)

    def _optimal_policy_rec(self, observations: ObservationSequence, actions: ActionSequence,
                            state_probs: jnp.ndarray, _seen_observations: dict[ObservationSequence, ActionSequence] | None = None,
                            _eps: float = 1e-10) -> tuple[PolicyDict, PolicyStats]:
        """Given a sequence of (observation, action, observation, action, ..., observation), and the hidden state probabilities,
        returns the optimal policy for this subtree, and the statistics (in particular EV and G)."""
        # print(f"{'  ' * len(observations)}Exploring observations {observations} and actions {actions}")

        assert len(observations) == len(actions)
        assert len(observations) <= self.time_horizon

        # Normalize state_probs (we already conditioned on the observations)
        state_probs = jnp.array(state_probs) / jnp.sum(state_probs)
        assert state_probs.shape == tuple(self.num_states), f"State probabilities shape {
            state_probs.shape} does not match expected {self.num_states}"

        # Overall policy and the policy and stats of each action
        stats = {"EV": 0.0, "G": 0.0, "nodes": 0}
        policy = {}

        # Compute the probabilities of *all* next observations in the current state
        op2s = [np.tensordot(self.A[i], state_probs, axes=self.num_factors)
                for i in range(self.num_modalities)]
        for i, op2 in enumerate(op2s):
            assert op2.shape == (self.num_obs[i],), f"Shape of observation[{i}] probabilities {
                op2.shape} does not match expected {self.num_obs[i]}"

        # Iterate over the possible observations
        for o in self.possible_observations:
            oprob = np.prod([op2s[i][o[i]]
                            for i in range(self.num_modalities)])
            # Check that the likelihood of this observation is larger than epsilon
            if oprob < _eps:
                # print(f"{'  ' * len(observations)}  - Skipping observation {o} after observations {observations} and actions {actions} with probability {oprob}")
                continue
            # print(f"{'  ' * len(observations)}  + Exploring observation {o} after observations {observations} and actions {actions} with probability {oprob}")

            observations2 = observations + (o,)
            if _seen_observations is not None:
                if observations2 in _seen_observations:
                    raise ValueError(f"Observation {observations2} already reached by actions {
                                     _seen_observations[observations2]}, reached again by {actions}.")
                _seen_observations[observations2] = actions

            # Update the state_probs conditioning on the observation
            evidence = np.prod([Af[o[i]]
                               for i, Af in enumerate(self.A)], axis=0)
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
                state_probs3 = np.tensordot(
                    B_agg2, state_probs2, axes=self.num_factors)
                assert np.isclose(
                    np.sum(state_probs3), 1.0, atol=1e-6), "Should still be a valid distribution"

                actions2 = actions + (a,)
                rec_policy, rec_stats = self._optimal_policy_rec(
                    observations2, actions2, state_probs3, _seen_observations=_seen_observations, _eps=_eps)
                # Collect the policy and update the stats
                a_policies[a] = rec_policy
                a_stats[a] = rec_stats

            # Select the action and therefore also policy for this observation
            best_a = min(a_stats, key=lambda a: a_stats[a]["G"])
            policy.update(a_policies[best_a])
            new_pol = np.zeros(self.num_controls)
            new_pol[*best_a] = 1.0
            policy[observations2] = new_pol
            stats = sum_dicts([stats, a_stats[best_a]], w=[
                              1.0, oprob], no_w_keys=["nodes"])

        # # Leaf observation had no actions
        # if len(observations2) > self.time_horizon:
        #     print(f"{'  ' * len(observations)}.. returning {stats}")
        #     return {}, stats

        # Select the best action
        # print(a_stats)
        # policy = a_policies[best_a]
        # print(f"{'  ' * len(observations)}.. selected {best_a}: {np.array_str(new_pol, max_line_width=100).replace('\n', ' ')}, returning {stats}")
        return policy, stats
