# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:46:14 2020

@author: hecro15
"""
import argparse
import gym
from gym.spaces import Box, Discrete
import minerl
import time
from datetime import date

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U
from baselines.common.tf_util import save_variables

from baselines import logger
from baselines import deepq
from baselines.deepq.models import build_q_func
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule
import cv2

class MineCraftWrapper:
    """Wrap the action and observation spaces of the MineCraft environment."""
    def __init__(self, minecraft_env):
        self.minecraft_env = minecraft_env
        self.action_space = Discrete(8)
        #Actions:
        #0: attack
        #1: back
        #2: camera left
        #3: camera right
        #4: forward
        #5: forward and jump
        #6: left
        #7: right
        self.observation_space = Box(low=0.0, high=1.0, shape=(64, 64, 2))
        #Observations (feature layers):
        #Grey scale image
        #compassAngle
        
    def step(self, action):
        minerl_action = self.action_to_minerl_action(action)
        minerl_obs, rew, done, info = self.minecraft_env.step(minerl_action)
        return self.minerl_obs_to_obs(minerl_obs), rew, done, info
        
    def action_to_minerl_action(self, action):
        minerl_action = self.minecraft_env.action_space.noop()
    
        if action==0:
            minerl_action['attack'] = 1
        elif action==1:
            minerl_action['back'] = 1
        elif action==2:
            minerl_action['camera'] = [0, 10.0]
        elif action==3:
            minerl_action['camera'] = [0, -10.0]
        elif action==4:
            minerl_action['forward'] = 1
        elif action==5:
            minerl_action['forward'] = 1
            minerl_action['jump'] = 1
        elif action==6:
            minerl_action['left'] = 1
        elif action==7:
            minerl_action['right'] = 1
            
        return minerl_action
        
    def minerl_obs_to_obs(self, minerl_obs):
        obs = np.ones(shape=(64,64,2))
        obs[:,:,0] = self.preprocess_image_frame(minerl_obs["pov"])
        obs[:,:,1] = obs[:,:,1] * ((minerl_obs["compassAngle"] + 180.0)/360.0)

        return obs
        
    def preprocess_image_frame(self, pov):
        """Preprocess image frame given from the environment."""
        grayscale_img = cv2.cvtColor(pov, cv2.COLOR_BGR2GRAY)
        grayscale_img = grayscale_img.astype('float64')/255.0
        return grayscale_img
        
    def reset(self):
        return self.minerl_obs_to_obs(self.minecraft_env.reset())

def q_network(input, num_actions, scope, reuse=False):
    """Build a neural network for the q function."""
    raise NotImplementedError

def train_policy(arglist):
    with U.single_threaded_session():
        # Create the environment
        env = gym.make('MineRLNavigate-v0')
        env = MineCraftWrapper(env)

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=build_q_func('conv_only', dueling=True),
            num_actions=env.action_space.n,
            gamma=0.9,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        # Create the replay buffer (TODO: Use prioritized replay buffer)
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
        log_path = "./learning_curves/minerl_" + str(date.today()) + "_" + str(time.time()) + ".dat"
        log_file = open(log_path, "a")
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
                n_episodes += 1

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if (n_steps > arglist.learning_starts_at_steps) and (n_steps % 4 == 0):
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

            # Update target network periodically.
            if n_steps % arglist.target_net_update_freq == 0:
                update_target()

            # Log data for analysis
            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", n_steps)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(n_steps)))
                logger.dump_tabular()
                print("%s,%s,%s,%s" % (n_steps,episode,round(np.mean(episode_rewards[-101:-1]), 1),int(100 * exploration.value(n_steps))), file=log_file)

            #TODO: Save checkpoints
            if episode % arglist.checkpoint_rate == 0:
                checkpoint_path = "./checkpoints/minerl_" + str(episode) + "_" + str(date.today()) + "_" + str(time.time()) + ".pkl"
                save_variables(checkpoint_path)
                
        log_file.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN policy.')
    parser.add_argument("--num-episodes", type=int, default=50000, help="number of episodes to use for training")
    parser.add_argument("--checkpoint-rate", type=int, default=10000, help="number of episodes to use for training")
    parser.add_argument("--replay-buffer-len", type=int, default=1000000, help="length of replay buffer")
    parser.add_argument("--num-exploration-steps", type=int, default=25000, help="number of time steps to use for exploration")
    parser.add_argument("--learning-starts-at-steps", type=int, default=10000, help="number of time steps before learning starts")
    parser.add_argument("--target-net-update-freq", type=int, default=1000, help="update frequency of the target network")
    parser.add_argument("--final-epsilon", type=float, default=0.02, help="final epsilon")
    parser.add_argument("--use-demonstrations", action="store_true", default=True)
    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    train_policy(arglist)
