from .multi_agent_env import MultiAgentEnv
import numpy as np
from typing import Any, Hashable, Iterable


class StochasticMultiAgentEnv(MultiAgentEnv):
    """
    Multi-agent environment defined by POMDP matrices (A_i, B, C_i, D)"""
    def __init__(self, As: Iterable[np.ndarray], B: np.ndarray, Cs: Iterable[np.ndarray], D: np.ndarray,
                 players:Iterable[str]|None=None, actions:Iterable[Iterable[Hashable]]| None=None,
                 observations:Iterable[Iterable[Hashable]]|None=None, states:Iterable[Hashable]|None = None,
                 rng_key:int|None=None):
        """Initialize a stochastic multi-agent environment by given matrices.
        
        Optionally provide player, actions, observations, and state values or names.
        Expected shapes are:
        - A[i][num_observations[i], num_states] - the observation probability matrix for player i
        - B[num_states_new, num_states_old, num_actions[0], num_actions[1], ...] - the transition probability matrix for the hidden state
        - C[i][num_observations[i]] - the observation-reward vector for player i
        - D[num_states] - the inital state prior distribution
        """
        self.As = tuple(As)
        self.B = B
        self.Cs = tuple(Cs)
        self.D = D

        if players is None:
            players = [f"P{i}" for i in range(len(self.As))]
        N = len(players)
        assert len(self.As) == N
        assert len(self.Cs) == N
        assert len(self.B.shape) == N + 2
        
        if observations is None:
            observations = [tuple(range(self.As[i].shape[0])) for i in range(len(players))]
        for i in range(N):
            assert self.As[i].shape[0] == len(observations[i])
            assert self.Cs[i].shape[0] == len(observations[i])

        if states is None:
            states = tuple(range(self.B.shape[0]))
        assert self.B.shape[0] == len(states)
        assert self.B.shape[1] == len(states)
        for i in range(N):
            assert self.As[i].shape[1] == len(states)
        assert self.D.shape[0] == len(states)

        if actions is None:
            actions = [tuple(range(self.B.shape[i+2])) for i in range(N)]
        for i in range(N):
            assert self.B.shape[i+2] == len(actions[i])

        super().__init__(players, actions, observations, states)
        self.prng = np.random.default_rng(rng_key)
        self.state = None

    def reset(self, rng_key:int|None=None) -> Iterable[int]:
        if rng_key is not None:
            self.prng = np.random.default_rng(rng_key)
        self.state = self.prng.choice(self.num_states, p=self.D)
        return [self.prng.choice(self.num_observations[i], p=self.As[i][:, self.state])
                for i in range(self.num_players)]
    
    def step(self, actions: Iterable[Hashable]) -> Iterable[int]:
        action_indexes = self._actions_to_indexes(actions)
        self.state = self.prng.choice(self.num_states, p=self.B[:, self.state, *action_indexes])
        return [self.prng.choice(self.num_observations[i], p=self.As[i][:, self.state])
                for i in range(self.num_players)]

    def rewards(self) -> Iterable[float]:
        return [self.Cs[i][self.state] for i in range(self.num_players)]


class OpenSpielEnv(StochasticMultiAgentEnv):
    """
    Multi-agent environment defined by OpenSpiel game.
    """
    def __init__(self, game: Any): # open_spiel.Game
        import open_spiel
        assert isinstance(game, open_spiel.Game)
        
        self.game = game
        self.num_players = game.num_players()
        self.players = tuple(f"P{i}" for i in range(self.num_players))
        self.actions = tuple(range(game.num_distinct_actions()))
        self.observations = tuple(range(game.num_distinct_observations()))
        self.states = tuple(range(game.num_states()))
        super().__init__(self.players, self.actions, self.observations, self.states)
        
        raise NotImplementedError("Not implemented yet")
