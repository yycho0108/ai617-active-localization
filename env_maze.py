#!/usr/bin/env python3

import gym
import time
import procgen
import cv2
import numpy as np
import torch as th
from abc import abstractmethod
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

_AGENT_COLOR: Tuple[int, int, int] = (127, 255, 127)
_WALL_COLOR: Tuple[int, int, int] = (191, 127, 63)
_WINDOW_SIZE: Tuple[int, int] = (7, 7)


class FlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(
                np.prod(_WINDOW_SIZE) * 3,), dtype=np.uint8)

    def observation(self, observation: np.ndarray):
        # return observation.ravel()
        return (observation.ravel() / 255.0) - 0.5


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(_WINDOW_SIZE + (3,)), dtype=np.uint8)
        # left, down, up, right, place-marker
        self._actions = [1, 3, 5, 7, -1]
        self._markers = []
        self._history = set()

    def _get_pos(self, obs: np.ndarray) -> Tuple[int, int]:
        return tuple(np.argwhere(np.equal(obs, _AGENT_COLOR).all(
            axis=-1)).mean(axis=0).astype(np.int32))

    def _add_markers(self, obs: np.ndarray) -> np.ndarray:
        if len(self._markers) <= 0:
            return obs
        for m in self._markers:
            obs[m[0], m[1]] = np.bitwise_or(obs[m[0], m[1]], (255, 0, 0))
        return obs

    def _down_obs(self, obs: np.ndarray):
        # NOTE(ycho): [15/64] only works for `easy`
        return cv2.resize(obs, None, fx=(15 / 64), fy=(15 / 64),
                          interpolation=cv2.INTER_NEAREST_EXACT)

    def _crop_obs(self, obs: np.ndarray, loc: Tuple[int, int]):
        # cv2.imshow('obs_', obs)
        # cv2.waitKey(1)
        # return obs
        radius: Tuple[int, int] = (_WINDOW_SIZE[0] // 2, _WINDOW_SIZE[1] // 2)
        out = np.full(_WINDOW_SIZE + (3,),
                      255,
                      dtype=obs.dtype)

        idx0 = (max(0, loc[0] - radius[0]), max(0, loc[1] - radius[1]))
        idx1 = (min(loc[0] + radius[0] + 1, obs.shape[0]),
                min(loc[1] + radius[1] + 1, obs.shape[1]))
        offset = np.subtract(idx0, (loc[0] - radius[0], loc[1] - radius[1]))
        roi = obs[idx0[0]:idx1[0], idx0[1]:idx1[1]]
        out[offset[0]:offset[0] + roi.shape[0],
            offset[1]:offset[1] + roi.shape[1], :] = roi
        return out

    def _recolor_obs(self, obs: np.ndarray):
        obs[(obs == _AGENT_COLOR).all(axis=-1)] = (0, 255, 0)
        obs[(obs == _WALL_COLOR).all(axis=-1)] = (0, 0, 255)
        return obs

    def reset(self):
        self._markers = []
        self._history = set()
        obs = self._down_obs(self.env.reset())
        loc = self._get_pos(obs)
        self._history.add(loc)
        return self._recolor_obs(self._crop_obs(obs, loc))

    def step(self, action: int):
        # Take action.
        if action == 4:
            # NOTE(ycho): 4 = "null action" in procgen.
            obs, rew, done, info = self.env.step(4)
        else:
            obs, rew, done, info = self.env.step(self._actions[action])

        # Process observations.
        info['obs0'] = obs
        obs = self._down_obs(obs)
        loc = self._get_pos(obs)

        # For marker-placement actions,
        # track marker placements and add
        # extra penalty for rewards.
        if action == 4:
            rew -= 0.01
            self._markers.append(loc)
        obs = self._add_markers(obs)

        # Penalize each timestep.
        rew -= 0.01

        # Especially penalize revisits.
        if loc in self._history:
            rew -= 0.05

        # Track history, format observation and
        # and return results.
        self._history.add(loc)
        return self._recolor_obs(self._crop_obs(
            obs, loc)), rew, done, info


class MazeEnv(gym.Env):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        env = gym.make('procgen:procgen-maze-v0',
                       # render_mode=render_mode,
                       # rand_seed=i,
                       start_level=0,
                       use_backgrounds=False,
                       use_monochrome_assets=True,
                       distribution_mode='easy',
                       # *args,
                       **env_config)
        env = CustomEnvWrapper(env)
        env = FlattenWrapper(env)
        self.env = env

        self.observation_space = gym.spaces.Box(
            low=-0.5, high=0.5, shape=(7 * 7 * 3,),
            dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


def make_env(*args, **kwds) -> gym.Env:
    env = gym.make('procgen:procgen-maze-v0',
                   # render_mode=render_mode,
                   # rand_seed=i,
                   start_level=0,
                   # center_agent=True,
                   use_backgrounds=False,
                   use_monochrome_assets=True,
                   distribution_mode='easy',
                   # distribution_mode='memory'
                   *args, **kwds)
    env = CustomEnvWrapper(env)
    env = FlattenWrapper(env)
    return env
