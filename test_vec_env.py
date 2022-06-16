#!/usr/bin/env python3

import cv2
import numpy as np
import einops
from vec_env_maze import MazeEnv, _WINDOW_SIZE


def main():
    num_envs: int = 8
    env = MazeEnv(dict(num_envs=num_envs))
    obs = env.reset()

    for _ in range(128):
        # for i, o in enumerate(obs):
        # o = o.reshape((7, 7, 3))[..., ::-1]
        # cv2.imshow(F'o{i}', (255 * (0.5 + o)).astype(np.uint8))
        vis = obs.reshape((num_envs * _WINDOW_SIZE[0], _WINDOW_SIZE[1], 3))
        cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
        cv2.imshow(F'vis', (255 * (0.5 + vis)).astype(np.uint8))
        cv2.waitKey(0)
        action = [env.action_space.sample() for _ in range(num_envs)]
        obs, rew, done, info = env.step(action)


if __name__ == '__main__':
    main()
