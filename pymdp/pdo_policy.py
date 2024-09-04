
import numpy as np
import jax.numpy as jnp
import jax

_Observation = tuple[int, ...]
_ObservationSequence = tuple[_Observation, ...]
_Action = tuple[int, ...]
_Policy = np.ndarray[np.float64]


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
        assert list(self.action_counts) == [
            len(a) for a in self.action_names], "action_counts must match action_names"

    def policy_for_observations(self, observations: _ObservationSequence) -> _Policy:
        """Given a series of observations, returns a JOINT distribution over actions, probabilities of shape self.action_counts"""
        raise NotImplementedError

    def sample_action_for_observations(self, observations: _ObservationSequence) -> _Action:
        """Given a series of observations, samples an action for each level of the hierarchy"""
        policy = self.policy_for_observations(observations)
        # Sample from the joint policy distribution, flattening it - return the indices of the sampled action
        return np.unravel_index(np.random.choice(self.n_actions, p=np.ravel(policy)), self.action_counts)

    @property
    def n_actions(self) -> int:
        return np.prod(self.action_counts)

    def uniform_policy(self) -> _Policy:
        return (np.ones(self.n_actions) / self.n_actions).reshape(self.action_counts)


class UniformPDOPolicy(PDOPolicyBase):
    def policy_for_observations(self, observations: _ObservationSequence) -> _Policy:
        return self.uniform_policy()


class TabularPolicy(PDOPolicyBase):
    """A TabularPolicy is a policy that is a table of probabilities for each action given a sequence of observations"""
    def __init__(self, *, action_counts: list[int] | None = None, action_names: list[list[str]] | None = None,
                 observation_sequences: list[_ObservationSequence], table: np.ndarray | jnp.ndarray | None = None):
        if action_counts is None and table is not None:
            action_counts = table.shape[1:]
        super().__init__(action_counts=action_counts, action_names=action_names)

        assert len(observation_sequences) == len(
            set(observation_sequences)), "Observation sequences must be unique"

        # convert observation sequences to tuples of tuples
        self.observation_sequences = tuple(
            tuple(tuple(observation) for observation in observation_seq) for observation_seq in observation_sequences)

        self.observation_seq_index = {
            obs: i for i, obs in enumerate(self.observation_sequences)}

        self.table = table
        if self.table is None:
            self.table = self._gen_default_table()
        assert len(observation_sequences) == self.table.shape[0], "Number of observation sequences must match table shape"

    def _gen_default_table(self) -> np.ndarray | jnp.ndarray:
        return jnp.ones(shape=(len(self.observation_sequences), *self.action_counts), dtype=jnp.float32) / self.n_actions

    def policy_for_observations(self, observation_sequence: _ObservationSequence) -> _Policy:
        obs = tuple(tuple(observation) for observation in observation_sequence)
        if obs not in self.observation_seq_index:
            raise ValueError(f"Observation sequence {obs} not in table")
        return self.table[self.observation_seq_index[obs]]
    
    def updated_copy(self, new_table: np.ndarray | jnp.ndarray) -> "TabularPolicy":
        return TabularPolicy(action_counts=self.action_counts, action_names=self.action_names,
                             observation_sequences=self.observation_sequences, table=new_table)

    @classmethod
    def from_dict(cls, policy_dict: dict[_ObservationSequence, _Action], action_names: list[list[str]] | None = None) -> "TabularPolicy":
        observation_sequences = sorted(policy_dict.keys())
        action_counts = policy_dict[observation_sequences[0]].shape
        print(action_counts)
        table = np.stack([policy_dict[obs] for obs in observation_sequences])
        return cls(action_counts=action_counts, action_names=action_names, observation_sequences=observation_sequences, table=table)

class TabularSoftmaxPolicy(TabularPolicy):
    """The softmax policy is a TabularPolicy with a softmax applied to the actions"""
    def policy_for_observations(self, observation_sequence: _ObservationSequence) -> _Policy:
        act = super().policy_for_observations(observation_sequence)
        return jax.nn.softmax(act, axis=None)

    def _gen_default_table(self) -> np.ndarray | jnp.ndarray:
        return jnp.zeros(shape=(len(self.observation_sequences), *self.action_counts), dtype=jnp.float32)

    def updated_copy(self, new_table: np.ndarray | jnp.ndarray) -> "TabularSoftmaxPolicy":
        return TabularSoftmaxPolicy(action_counts=self.action_counts, action_names=self.action_names,
                                    observation_sequences=self.observation_sequences, table=new_table)


## Register TabularPolicy with Jax tree_util to be passed as a parameter
jax.tree_util.register_pytree_node(
    TabularPolicy,
    lambda p: ((p.table,), (p.action_counts, p.action_names, p.observation_sequences)),
    lambda x, y: TabularPolicy(action_counts=x[0], action_names=x[1], observation_sequences=x[2], table=y[0])
)


## Register TabularSoftmaxPolicy with Jax tree_util to be passed as a parameter
jax.tree_util.register_pytree_node(
    TabularSoftmaxPolicy,
    lambda p: ((p.table,), (p.action_counts, p.action_names, p.observation_sequences)),
    lambda x, y: TabularSoftmaxPolicy(action_counts=x[0], action_names=x[1], observation_sequences=x[2], table=y[0])
)
