#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" EV and PDO agents - G-minimization via gradient descent

__author__: Tomáš Gavenčiak

"""

import numpy as np
import jax.numpy as jnp
from .full_policy import FullPolicyBase
from .common import outer_product
from .agent_base import BranchingAgent


class PDOAgentGradient(BranchingAgent):
    """Agent minimizing PDO"""

    def G(self, policy: FullPolicyBase):
        "This needs to be JAX-differentiable wrt values from `policy.table`"
        B_agg = self.compute_B_agg(with_jax=True)
        uniform_policy = policy.uniform_policy()

        def _helper(observations: tuple[tuple[int]], state_prob: jnp.ndarray) -> jnp.ndarray:
            if len(observations) > self.time_horizon:
                return 0.0

            FE = jnp.array(0.0)
            for o in self.possible_observations:
                # This couan be just an numpy product - does not depend on policy
                evidence = jnp.prod([Af[o[i]]
                                    for i, Af in enumerate(self.A)], axis=0)

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
                FE += (1 / self.beta) * jnp.sum(sb2) * jnp.sum(po2 *
                                                               (jnp.log2(po2 + 1e-10) - jnp.log2(ppo2 + 1e-10)))

                # Apply policy actions to B_agg
                B_agg3 = jnp.tensordot(B_agg, po2, axes=self.num_factors)
                # Apply previous state belief to B3
                sb4 = jnp.tensordot(B_agg3, sb2, axes=self.num_factors)
                FE += _helper(obs2, sb4)

            return FE

        return _helper((), outer_product(*self.D))


class EVAgentGradient(BranchingAgent):
    """Agent minimizing expected value of the reward (sum over all the timesteps)"""

    def G(self, policy: FullPolicyBase):
        B_agg = self.compute_B_agg(with_jax=True)

        def _helper(observations: tuple[tuple[int]], state_prob: jnp.ndarray) -> jnp.ndarray:
            if len(observations) > self.time_horizon:
                return 0.0

            FE = jnp.array(0.0)
            for o in self.possible_observations:
                # This can be just NP product - does not depend on policy
                evidence = np.prod([Af[o[i]]
                                   for i, Af in enumerate(self.A)], axis=0)

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
