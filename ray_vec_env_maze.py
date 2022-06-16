#!/usr/bin/env python3

import logging
from typing import Dict, Any, Optional
from vec_env_maze import MazeEnv, _WINDOW_SIZE
from ray.rllib.env.vector_env import VectorEnv
# from ray.rllib.env.remote_vector_env import RemoteVectorEnv


class RayMazeEnv(VectorEnv):
    def __init__(self, env_config: Dict[str, Any] = None):
        if env_config is None:
            env_config = {'num_envs': 1}
        num_envs = env_config.pop('num_envs')

        self.envs = MazeEnv({**env_config, 'num_envs': num_envs})
        super().__init__(self.envs.observation_space,
                         self.envs.action_space,
                         num_envs)
        self._prv_obs = [None for _ in range(num_envs)]

    def reset_at(self, index: Optional[int] = None):
        return self._prv_obs[index]

    def vector_reset(self):
        obs = self.envs.reset()
        self._prv_obs = obs
        return obs

    def vector_step(self, actions):
        obs, rew, done, info = self.envs.step(actions)
        self._prv_obs = obs
        return obs, rew, done, info
