#!/usr/bin/env python3

import gym
import cv2
import numpy as np
from pathlib import Path
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer
from typing import Dict

from env_maze import MazeEnv
from ray_vec_env_maze import RayMazeEnv

from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import torch as th
from vis import plot_categorical_distribution, obs2vis


def get_config(**kwds):
    out = {
        'framework': 'torch',

        'env': RayMazeEnv,
        'env_config': {
           'num_envs': 16
        },
        # 'env': MazeEnv,

        # set `num_workers=0` when using ICM
        'num_workers': 8,
        'model': {
            'use_lstm': False
        },
        'explore': True,
        'exploration_config': {
            'type': 'StochasticSampling'
            #'type': 'Curiosity',
            #'eta': 1.0,
            #'lr': 0.001,
            #'feature_dim': 288,
            #"feature_net_config": {
            #    "fcnet_hiddens": [],
            #    "fcnet_activation": "relu",
            #},
            #"inverse_net_hiddens": [256],
            #"inverse_net_activation": "relu",
            #"forward_net_hiddens": [256],
            #"forward_net_activation": "relu",
            #"beta": 0.2,
            #"sub_exploration": {
            #    "type": "StochasticSampling",
            #}
        },

        'gamma': 0.998,

        'train_batch_size': 2048,
        'sgd_minibatch_size': 2048,
        'rollout_fragment_length': 64,
        'num_sgd_iter': 3,
        'lr': 5e-5,

        'vf_loss_coeff': 0.5,
        'vf_share_layers': True,
        'kl_coeff': 0.0,
        'kl_target': 0.1,
        'clip_param': 0.1,
        'entropy_coeff': 0.005,

        'grad_clip': 1.0,
        'lambda': 0.8,

        'num_gpus': 1,
    }
    out.update(kwds)
    return out


def train():
    trainer = PPOTrainer(
        config=get_config()
    )

    # ckpt_path = '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-08_23-07-07h_zgmyyd/checkpoint_008192/checkpoint-8192'
    # ckpt_path = '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-08_23-07-07h_zgmyyd/checkpoint_008192/checkpoint-8192'
    # ckpt_path = '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-15_00-08-22oalwl0qd/checkpoint_016384/checkpoint-16384'
    # trainer.restore(ckpt_path)

    try:
        with tqdm(range(32768)) as pbar:
            for i in pbar:
                results = trainer.train()
                if i % 64 == 0:
                    avg_reward = results['episode_reward_mean']
                    pbar.set_description(
                        F'Iter: {i}; avg.rew={avg_reward:02f}')
                if i % 1024 == 0:
                    ckpt = trainer.save()
                    print(F'saved ckpt = {ckpt}')
    finally:
        ckpt = trainer.save()
        print(F'saved ckpt = {ckpt}')


def test():
    #ckpt_path = (
    #    '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-07_20-03-21uf2qyiot/checkpoint_000769/checkpoint-769')
    # ckpt_path = '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-07_20-43-58p2b_uojo/checkpoint_004096/checkpoint-4096'
    # ckpt_path = '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-07_21-24-362rc0x9tk/checkpoint_004096/checkpoint-4096'
    # ckpt_path = '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-08_09-43-24mwhurdfh/checkpoint_004096/checkpoint-4096'
    # ckpt_path = '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-08_17-22-466t71fx75/checkpoint_008192/checkpoint-8192'
    # ckpt_path = '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-09_01-07-334l0_nzrp/checkpoint_024576/checkpoint-24576'
    ckpt_path = '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-15_04-27-14f_nxdd7a/checkpoint_049152/checkpoint-49152'
    config = get_config(num_workers=0)
    agent = PPOTrainer(config=config,
                       env=MazeEnv)
    use_lstm: bool = config.get('model', {}).get('use_lstm', False)
    env = MazeEnv(dict(render_mode='human'))
    agent.restore(ckpt_path)

    done: bool = True
    if use_lstm:
        state = agent.get_policy().get_initial_state()
    prev_action = None
    prev_reward = None
    obss = []
    steps = 0
    sav = True
    probs = []
    while True:
        if done:
            obs = env.reset()
            if use_lstm:
                state = agent.get_policy().get_initial_state()
            prev_action = None
            prev_reward = None
            obss = []
            sav = True
            plot_categorical_distribution(probs, steps=steps)
            probs = []

        if use_lstm:
            action, state, extra = agent.compute_single_action(
                obs, state, explore=False)  # obs, state_in=state))
            probs.append(
                th.softmax(
                    th.as_tensor(extra['action_dist_inputs']),
                    dim=- 1).ravel().detach().cpu().numpy())
        else:
            action = agent.compute_single_action(obs, explore=False)
        if action == 4:
            print(F'Took {steps} steps')
            break
        obs, reward, done, info = env.step(action)

        obss.append(info['obs0'])
        steps += 1
        if steps % 1000 == 0:
            print(steps)

        # Show `obs`.
        cv2.namedWindow('obs', cv2.WINDOW_NORMAL)
        cv2.imshow('obs', (obs + 0.5).reshape((7, 7, 3))[..., ::-1])
        cv2.waitKey(1)

        if True:
            vis = cv2.resize(
                (obs + 0.5).reshape(7, 7, 3)[..., :: -1],
                dsize=None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
            if sav:
                cv2.imwrite('/tmp/obs.png', (vis * 255).astype(np.uint8))
                cv2.imwrite('/tmp/obs0.png', info['obs0'])
                sav = False

                q = input('quit?')
                if len(q) >= 1 and q[0] == 'y':
                    break

        #if reward >= 9.8:
        #    for ii, o in enumerate(obss):
        #        cv2.imwrite(F'/tmp/suc-obs/{ii:03d}.png', o)
        #    q = input('quit?')
        #    if len(q) >= 1 and q[0] == 'y':
        #        break

        # print('obs', obs.shape)
        #cv2.namedWindow('obs', cv2.WINDOW_NORMAL)
        #vis = cv2.resize((obs + 0.5).reshape(7, 7, 3)[..., ::-1], dsize=None,
        #                 fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        #cv2.imshow('obs', vis)
        #cv2.waitKey(5)
        # print('reward', reward)

        prev_action = action
        prev_reward = reward


def main():
    train()
    # test()


if __name__ == '__main__':
    main()
