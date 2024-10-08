#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Common types and code for PDO-based agents

__author__: Tomáš Gavenčiak

"""


import functools
from typing import Iterable
import jax
import numpy as np
import jax.numpy as jnp


Observation = tuple[int, ...]
ObservationSequence = tuple[Observation, ...]
Action = tuple[int, ...]
ActionSequence = tuple[Action, ...]
ActionDistribution = np.ndarray[np.float64]
PolicyDict = dict[ObservationSequence, ActionDistribution]
PolicyStats = dict[str, int | float | np.ndarray]


def sum_dicts(dicts: Iterable[dict], w: Iterable[float] | None = None, *, no_w_keys=()) -> dict:
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
        return functools.reduce(lambda x, y: jnp.tensordot(x, y, axes=([], [])), arrays)
    elif isinstance(arrays[0], np.ndarray):
        return functools.reduce(lambda x, y: np.tensordot(x, y, axes=([], [])), arrays)
    else:
        raise ValueError(f"Arrays must be either jnp.ndarray or np.ndarray, got {
                         type(arrays[0])}")
