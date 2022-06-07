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


class FlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(7 * 7 * 3,), dtype=np.uint8)

    def observation(self, observation: np.ndarray):
        # return observation.ravel()
        return (observation.ravel() / 255.0) - 0.5


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)
        # left, down, up, right
        self._actions = [1, 3, 5, 7, -1]
        self._markers = []

    def _get_pos(self, obs: np.ndarray) -> Tuple[int, int]:
        return np.argwhere(np.equal(obs, _AGENT_COLOR).all(
            axis=-1)).mean(axis=0).astype(np.int32)

    def _add_markers(self, obs: np.ndarray) -> np.ndarray:
        if len(self._markers) <= 0:
            return obs
        for m in self._markers:
            obs[m[0], m[1]] = (255, 0, 0)
        return obs

    def _crop_obs(self, obs: np.ndarray, loc: Tuple[int, int]):
        # return obs
        window_size: Tuple[int, int] = (7, 7)
        radius: Tuple[int, int] = (window_size[0] // 2, window_size[1] // 2)
        out = np.zeros(window_size + (3,), dtype=obs.dtype)

        idx0 = (max(0, loc[0] - radius[0]), max(0, loc[1] - radius[1]))
        idx1 = (min(loc[0] + radius[0] + 1, obs.shape[0]),
                min(loc[1] + radius[1] + 1, obs.shape[1]))
        offset = np.subtract(idx0, (loc[0] - radius[0], loc[1] - radius[1]))
        roi = obs[idx0[0]:idx1[0], idx0[1]:idx1[1]]
        out[offset[0]:offset[0] + roi.shape[0],
            offset[1]:offset[1] + roi.shape[1], :] = roi
        return out

    def reset(self):
        self._markers = []
        obs = self.env.reset()
        loc = self._get_pos(obs)
        return self._crop_obs(obs, loc)

    def step(self, action: int):
        if action == 4:
            # NOTE(ycho): 4 = null action in procgen.
            obs, rew, done, info = self.env.step(4)
            loc = self._get_pos(obs)
            self._markers.append(loc)
            obs = self._add_markers(obs)
            # NOTE(ycho): override reward:-1
            return self._crop_obs(obs, loc), -1, done, info
        obs, rew, done, info = self.env.step(self._actions[action])
        obs = self._add_markers(obs)
        loc = self._get_pos(obs)
        return self._crop_obs(obs, loc), rew - 0.1, done, info


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
