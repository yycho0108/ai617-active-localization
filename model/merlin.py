#!/usr/bin/env python3

from abc import abstractmethod, ABC
import torch as th


class AgentBase(nn.Module):
    @abstractmethod
    def reset_state(state: th.Tensor = None):
        pass

    @abstractmethod
    def get_state():
        pass

    @abstractmethod
    def add_obs(obs: th.Tensor):
        """Add observation."""
        pass

    @abstractmethod
    def get_action_distribution(state) -> th.Tensor:
        pass

    @abstractmethod
    def get_action(state) -> th.Tensor:
        pass


class Learner:
    def __init__(self, agent):
        self.agent = agent


class Merlin(AgentBase):
    def __init__(self):
        super().__init__()

    def reset_state(state: th.Tensor = None):
        self.state = state
