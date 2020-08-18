# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:46:14 2020

@author: hecro15
"""
import argparse
import gym
import minerl

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

class MineCraftWrapper:
    """Wrap the action and observation spaces of the MineCraft environment."""
    def __init__(self, minecraft_env):
        self.minecraft_env = minecraft_env

def preprocess_image_frame():
    """Preprocess image frame given from the environment."""
    raise NotImplementedError

def q_network(input, num_actions, scope, reuse=False):
    """Build a neural network for the q function."""
    raise NotImplementedError

def train_policy(arglist):
    with U.make_session(num_cpu=8):
        env = gym.make('MineRLNavigate-v0')
        env = MineCraftWrapper(env)
        # Create all the functions necessary to train the model
        # TODO: Add arguments for train hyper parameters
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=q_network,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(arglist.replay_buffer_len)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=arglist.num_exploration_steps, initial_p=1.0, final_p=arglist.final_epsilon)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        n_episodes = 0
        n_steps = 0
        obs = env.reset()
        for episode in range(arglist.num_episodes):
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(n_steps))[0]
            new_obs, rew, done, _ = env.step(action)
            n_steps += 1
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if n_steps > arglist.learning_starts_at_steps:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            # Update target network periodically.
            if n_steps % arglist.target_net_update_freq == 0:
                update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", n_steps)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(n_steps)))
                logger.dump_tabular()

            #TODO: Save checkpoints

def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN policy.')
    parser.add_argument("--num-episodes", type=int, default=50000, help="number of episodes to use for training")
    parser.add_argument("--replay-buffer-len", type=int, default=1000000, help="length of replay buffer")
    parser.add_argument("--num-exploration-steps", type=int, default=25000, help="number of time steps to use for exploration")
    parser.add_argument("--learning-starts-at-steps", type=int, default=10000, help="number of time steps before learning starts")
    parser.add_argument("--target-net-update-freq", type=int, default=1000, help="update frequency of the target network")
    parser.add_argument("--final-epsilon", type=float, default=0.02, help="final epsilon")
    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    train_policy(arglist)
