#!/usr/bin/env python3

import gym
import time
import procgen
import cv2
import numpy as np
import torch as th
from abc import abstractmethod
from typing import Tuple, Optional, Dict, Any, Set, List
from pathlib import Path
import einops

_AGENT_COLOR: Tuple[int, int, int] = (127, 255, 127)
_WALL_COLOR: Tuple[int, int, int] = (191, 127, 63)
_WINDOW_SIZE: Tuple[int, int] = (7, 7)
_WINDOW_RAD: Tuple[int, int] = (
    _WINDOW_SIZE[0] // 2, _WINDOW_SIZE[1] // 2)


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


class EnvState:
    def __init__(self):
        self.markers: List[Tuple[int, int]] = []
        self.history: Set[Tuple[int, int]] = set()

    def reset(self):
        self.markers = []
        self.history = set()

    def step(self, loc: Tuple[int, int],
             mark: bool = True):
        if mark:
            self.markers.append(loc)
        self.history.add(loc)


def _crop_obs(obs: np.ndarray, loc: Tuple[int, int]) -> np.ndarray:
    radius: Tuple[int, int] = _WINDOW_RAD
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


def convert_obs(obs: np.ndarray):
    radius: Tuple[int, int] = (
        _WINDOW_SIZE[0] // 2, _WINDOW_SIZE[1] // 2)

    # Downsample
    # FIXME(ycho): this only works for `easy` distribution
    # which has map size of (15x15).

    # temporarily put batch dim to height dimension.
    shape = obs.shape
    H = obs.shape[-3]
    obs = einops.rearrange(obs, '... h w c -> (... h) w c')
    N = obs.shape[0] // H
    obs = cv2.resize(obs, (15, N * 15),
                     interpolation=cv2.INTER_NEAREST_EXACT)

    # Figure out agent location.
    obs = obs.reshape(shape[:-3] + (15, 15, 3))
    loc = np.argwhere(np.equal(obs, _AGENT_COLOR).all(axis=-1))[..., -2:]

    # Crop.
    out = []
    for oi, li in zip(obs, loc):
        oi = _crop_obs(oi, li)
        #cv2.namedWindow(F'oi-{oi}', cv2.WINDOW_NORMAL)
        #cv2.imshow(F'oi-{oi}', oi)
        out.append(oi)
    # cv2.waitKey(0)
    obs = np.stack(out, axis=0)

    # Recolor.
    obs[(obs == _AGENT_COLOR).all(axis=-1)] = (0, 255, 0)
    obs[(obs == _WALL_COLOR).all(axis=-1)] = (0, 0, 255)

    # Return.
    return (obs, loc)


class MazeEnv(gym.Env):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        num_envs: int = env_config.pop('num_envs', 1)
        self.num_envs = num_envs
        self.env = procgen.ProcgenEnv(num_envs,
                                      'maze',
                                      start_level=0,
                                      use_backgrounds=False,
                                      use_monochrome_assets=True,
                                      distribution_mode='easy',
                                      **env_config)

        self.observation_space = gym.spaces.Box(
            low=-0.5, high=0.5, shape=(7 * 7 * 3,),
            dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)

        # left, down, up, right, place-marker
        self._states = [EnvState() for _ in range(num_envs)]
        self._actions = np.asanyarray([1, 3, 5, 7, 4],
                                      dtype=np.int32)
        self._prev_actions = None

    def _flatten_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = einops.rearrange(obs, '... h w c -> ... (h w c)')
        return (obs / 255.0) - 0.5

    def reset(self):
        obs = self.env.reset()
        obs, loc = convert_obs(obs['rgb'])
        obs = self._flatten_obs(obs)
        return obs

    def step(self, action: np.ndarray):
        self.step_async(action)
        return self.step_wait()

    def step_async(self, action: np.ndarray):
        action = np.asanyarray(action, dtype=np.int32)
        # convert action space.
        action = self._actions[action]
        # save prev action
        self._prev_actions = action
        # step async.
        return self.env.step_async(action)

    def step_wait(self):
        obs, rew, done, info = self.env.step_wait()
        obs, loc = convert_obs(obs['rgb'])
        actions = self._prev_actions

        for i in range(self.num_envs):
            # Advance states.
            self._states[i].step(tuple(loc[i]),
                                 actions[i] == 4)

            # Reset all env-states for completed envs.
            if done[i]:
                self._states[i].reset()

            # Augment observation with markers.
            for m in self._states[i].markers:
                m = np.subtract(m, loc[i]) + _WINDOW_RAD
                if (m < 0).any() or (m >= _WINDOW_SIZE).any():
                    continue
                obs[i, m[0], m[1]] = np.bitwise_or(
                    obs[i, m[0], m[1]], (255, 0, 0))

        obs = self._flatten_obs(obs)
        rew -= 0.01
        return (obs, rew, done, info)


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
