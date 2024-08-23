
import numpy as np
import jax.numpy as jnp
import jax

Observation = tuple[int]
ObservationSequence = tuple[Observation]
Action = tuple[int]
Policy = np.ndarray[np.float64]


class PDOPolicyBase:
    def __init__(self, *, action_counts: list[int] | None = None, action_names: list[list[str]] | None = None):
        self.action_counts = action_counts
        self.action_names = action_names

        if action_counts is None:
            assert action_names is not None, "If action_counts is None, action_names must be provided"
            self.action_counts = [len(a) for a in action_names]
            self.action_names = action_names
        elif action_names is None:
            assert action_counts is not None, "If action_names is None, action_counts must be provided"
            self.action_counts = action_counts
            self.action_names = [
                [f"A{n}-{i}" for i in range(n)] for n in action_counts]
        assert action_counts == [
            len(a) for a in action_names], "action_counts must match action_names"

    def policy_for_observations(self, observations: ObservationSequence) -> Policy:
        """Given a series of observations, returns a JOINT distribution over actions, probabilities of shape self.action_counts"""
        raise NotImplementedError

    def sample_action_for_observations(self, observations: ObservationSequence) -> Action:
        """Given a series of observations, samples an action for each level of the hierarchy"""
        policy = self.policy_for_observations(observations)
        # Sample from the joint policy distribution, flattening it - return the indices of the sampled action
        return np.unravel_index(np.random.choice(range(self.n_actions), p=policy.flatten()), self.action_counts)

    @property
    def n_actions(self) -> int:
        return np.prod(self.action_counts)

    def uniform_policy(self) -> Policy:
        return (np.ones(self.n_actions) / self.n_actions).reshape(self.action_counts)


class UniformPDOPolicy(PDOPolicyBase):
    def policy_for_observations(self, observations: ObservationSequence) -> Policy:
        return self.uniform_policy()


class TabularPolicy(PDOPolicyBase):
    def __init__(self, *, action_counts: list[int] | None = None, action_names: list[list[str]] | None = None,
                 observation_sequences: list[ObservationSequence], table: np.ndarray | jnp.ndarray | None = None):
        super().__init__(action_counts=action_counts, action_names=action_names)

        self.observation_sequences = tuple(
            tuple(tuple(observation) for observation in observation_seq) for observation_seq in observation_sequences)

        self.observation_seq_index = {
            obs: i for i, obs in enumerate(self.observation_sequences)}

        self.table = table
        if self.table is None:
            self.table = jnp.ones(
                shape=(len(self.observation_sequences), *self.action_counts), dtype=np.float64) / self.n_actions

    def policy_for_observations(self, observation_sequence: ObservationSequence) -> Policy:
        obs = tuple(tuple(observation) for observation in observation_sequence)
        if obs not in self.observation_seq_index:
            raise ValueError(f"Observation sequence {obs} not in table")
        return self.table[self.observation_seq_index[obs]]
    
    def updated_copy(self, new_table: np.ndarray | jnp.ndarray) -> "TabularPolicy":
        return TabularPolicy(action_counts=self.action_counts, action_names=self.action_names,
                             observation_sequences=self.observation_sequences, table=new_table)

## Register TabularPolicy with Jax tree_util to be passed as a parameter
jax.tree_util.register_pytree_node(
    TabularPolicy,
    lambda p: ((p.action_counts, p.action_names, p.observation_sequences), p.table),
    lambda x, y: TabularPolicy(*x, table=y)
)