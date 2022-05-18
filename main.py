#!/usr/bin/env python3

import gym
import time
import procgen
import cv2
import numpy as np
import torch as th
from abc import abstractmethod
from typing import Tuple, Optional
from pathlib import Path
from functools import partial
from tqdm.auto import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy


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
        agent_color: Tuple[int, int, int] = (127, 255, 127)
        return np.argwhere(np.equal(obs, agent_color).all(
            axis=-1)).mean(axis=0).astype(np.int32)

    def _add_markers(self, obs: np.ndarray) -> np.ndarray:
        if len(self._markers) <= 0:
            return obs
        for m in self._markers:
            obs[m[0], m[1]] = (255, 0, 0)
        return obs

    def _crop_obs(self, obs: np.ndarray, loc: Tuple[int, int]):
        return obs
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
        return self._crop_obs(obs, loc), rew, done, info


def test_env():
    env = gym.make('procgen:procgen-maze-v0',
                   # render_mode='human',
                   # rand_seed=0,
                   start_level=0,
                   center_agent=True,
                   use_backgrounds=False,
                   use_monochrome_assets=True,
                   # distribution_mode='memory'
                   )

    env = CustomEnvWrapper(env)

    done: bool = True
    prv_obs = None
    Path('/tmp/ai617').mkdir(parents=True, exist_ok=True)
    step: int = 0
    while True:
        # print(env.action_space.sample())
        if done:
            prv_obs = env.reset()
        action = env.action_space.sample()
        if action == 4 and np.random.uniform() < 0.9:
            action = np.random.randint(4)
        obs, rew, done, info = env.step(action)
        prv_obs = obs

        cv2.namedWindow('obs', cv2.WINDOW_NORMAL)
        vis = cv2.resize(obs[...,::-1], dsize=None,
                fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(F'/tmp/ai617/obs-{step:03d}.png', vis)
        # cv2.imshow('obs', obs[..., ::-1])
        # cv2.waitKey(10)

        # env.render()
        print(f"step {step} reward {rew} done {done}")
        step += 1
        if done:
            break


class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])

    def _on_step(self):
        self.progress_bar.update(
            self.num_timesteps - self.progress_bar.n
        )
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None


class TensorboardGraphCallback(BaseCallback):
    def __init__(self, dummy: th.Tensor, *args, **kwds):
        super().__init__(*args, **kwds)
        self._dummy = dummy

    def _on_training_start(self):
        ofs = self.logger.output_formats
        tbf = next(f for f in ofs if isinstance(f, TensorBoardOutputFormat))
        tbf.writer.add_graph(
            self.model.policy,
            th.as_tensor(
                self._dummy)[None].to(
                self.model.device))
        tbf.writer.flush()

    def _on_step(self):
        return True


def main():
    path: str = '/tmp/ai617/train'
    train: bool = True
    num_env = 8 if train else 1
    render_mode = None if train else 'human'
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'env_log').mkdir(parents=True, exist_ok=True)
    (path / 'tb_log').mkdir(parents=True, exist_ok=True)
    (path / 'ckpt').mkdir(parents=True, exist_ok=True)
    (path / 'eval').mkdir(parents=True, exist_ok=True)

    def make_env(log_root: str, i: int, render_mode: Optional[str] = None):
        env = gym.make('procgen:procgen-maze-v0',
                       render_mode=render_mode,
                       rand_seed=i,
                       start_level=0,
                       # center_agent=True,
                       use_backgrounds=False,
                       use_monochrome_assets=True,
                       distribution_mode='easy',
                       # distribution_mode='memory'
                       )
        env = CustomEnvWrapper(env)
        env = FlattenWrapper(env)
        log_file = str(Path(log_root) / F'{i:02d}')
        env = Monitor(env, log_file)
        return env

    env = SubprocVecEnv([partial(make_env, str(path / 'env_log'), i,
                                 render_mode=render_mode)
                         for i in range(num_env)])

    #agent = PPO('MlpPolicy', env, verbose=0,
    #            n_steps=128,
    #            batch_size=32,
    #            tensorboard_log=path / 'tb_log',
    #            create_eval_env=False)

    agent = RecurrentPPO(MlpLstmPolicy, env, verbose=0,
                         n_steps=128,
                         batch_size=32,
                         tensorboard_log=path / 'tb_log',
                         create_eval_env=False)

    # if load:
    # agent.load('/tmp/ai617/train/ckpt/rl_model_96000_steps.zip')

    if train:
        checkpoint_callback = CheckpointCallback(
            save_freq=4000, save_path=path / 'ckpt',
            name_prefix="rl_model")
        tb_graph_callback = TensorboardGraphCallback(
            env.observation_space.sample())
        tqdm_callback = TqdmCallback()

        agent.learn(total_timesteps=100000,
                    eval_freq=4, n_eval_episodes=8,
                    eval_log_path=path / 'eval',
                    log_interval=4,
                    callback=[checkpoint_callback, tqdm_callback,
                              # tb_graph_callback
                              ])
    else:
        # eval
        obs = env.reset()
        state = None
        while True:
            action, state = agent.predict(obs, state)
            obs, _, done, _ = env.step(action)
            # env.render()
            if np.any(done):
                break


if __name__ == '__main__':
    test_env()
