#!/usr/bin/env python3

import gym
from pathlib import Path
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer
from typing import Dict
from env_maze import MazeEnv
from tqdm.auto import tqdm


def get_config(**kwds):
    out = {
        'framework': 'torch',
        'env': MazeEnv,
        'num_workers': 8,
        'model': {
            'use_lstm': False
        },
        'gamma': 0.99,
        'train_batch_size': 256,
        'num_gpus': 1,
    }
    out.update(kwds)
    return out


def train():
    trainer = PPOTrainer(
        config=get_config()
    )

    try:
        with tqdm(range(4096)) as pbar:
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
    ckpt_path = '/home/jamiecho/ray_results/PPOTrainer_MazeEnv_2022-06-07_21-24-362rc0x9tk/checkpoint_004096/checkpoint-4096'
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
    while True:
        if done:
            obs = env.reset()
            if use_lstm:
                state = agent.get_policy().get_initial_state()
            prev_action = None
            prev_reward = None

        if use_lstm:
            action, state, _ = agent.compute_single_action(
                obs, state, explore=None)  # obs, state_in=state))
        else:
            action = agent.compute_single_action(obs, explore=False)
        obs, reward, done, info = env.step(action)
        print('reward', reward)

        prev_action = action
        prev_reward = reward


def main():
    # train()
    test()


if __name__ == '__main__':
    main()
