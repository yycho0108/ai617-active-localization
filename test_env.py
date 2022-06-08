#!/usr/bin/env python3

import cv2
import time
from env_maze import make_env, MazeEnv


def main():
    env = MazeEnv(dict(render_mode=None))
    done: bool = False
    for _ in range(1024):
        if done:
            prv_obs = env.reset()
        obs, rew, done, info = env.step(env.action_space.sample())

        cv2.namedWindow('obs', cv2.WINDOW_NORMAL)
        cv2.imshow('obs', (obs + 0.5).reshape((7, 7, 3))[..., ::-1])
        cv2.waitKey(1)

        prv_obs = obs
        time.sleep(0.01)


if __name__ == '__main__':
    main()
