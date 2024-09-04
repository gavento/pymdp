#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A module with PDO-based agents

__author__: Tomáš Gavenčiak

"""

from . import agent_base, agent_direct, agent_gradient, common
from .agent_direct import EVAgentDirect
from .agent_gradient import EVAgentGradient, PDOAgentGradient
