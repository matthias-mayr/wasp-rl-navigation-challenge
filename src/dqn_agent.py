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
import random

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
from baselines.common.atari_wrappers import FrameStack
import cv2

import psutil
import sys


class MineCraftWrapper:
    """Wrap the action and observation spaces of the MineCraft environment."""
    def __init__(self, minecraft_env):
        self.minecraft_env = minecraft_env
        self.reward_range = np.array([-100,100])
        self.metadata = {}
        self.action_space = Discrete(9)
        #Actions:
        #0: attack
        #1: back
        #2: camera left
        #3: camera right
        #4: forward
        #5: forward and jump
        #6: left
        #7: right
        #8: no-op
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

    def minerl_action_to_action(self, minerl_action):
        actions = []

        if minerl_action['attack'] == 1:
            actions.append(0)
        elif minerl_action['back'] == 1:
            actions.append(1)
        elif minerl_action['camera'][1] > 10.0:
            actions.append(2)
        elif minerl_action['camera'][1] < -10.0:
            actions.append(3)
        elif minerl_action['forward'] == 1:
            actions.append(4)
        elif minerl_action['forward'] == 1 and minerl_action['jump'] == 1:
            actions.append(5)
        elif minerl_action['left'] == 1:
            actions.append(6)
        elif minerl_action['right'] == 1:
            actions.append(7)

        if len(actions) > 0:
            action = random.choice(actions)
        else:
            action = 8
            
        return action

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

class MineCraftWrapperSimplified:
    """Wrap the action and observation spaces of the MineCraft environment."""
    def __init__(self, minecraft_env):
        self.minecraft_env = minecraft_env
        self.reward_range = np.array([-100,100])
        self.metadata = {}
        self.action_space = Discrete(5)
        #Actions:
        #0: attack
        #1: back
        #2: camera left
        #3: camera right
        #4: forward
        #5: forward and jump
        #6: left
        #7: right
        #8: no-op
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
        minerl_action['attack'] = 1
        minerl_action['forward'] = 1
        minerl_action['jump'] = 1
    
        if action==0:
            minerl_action['camera'] = [0, 10.0]
        elif action==1:
            minerl_action['camera'] = [0, -10.0]
        elif action==2:
            minerl_action['left'] = 1
        elif action==3:
            minerl_action['right'] = 1
            
        return minerl_action

    def minerl_action_to_action(self, minerl_action):
        actions = []

        if minerl_action['camera'][1] > 10.0:
            actions.append(2)
        elif minerl_action['camera'][1] < -10.0:
            actions.append(3)
        elif minerl_action['left'] == 1:
            actions.append(6)
        elif minerl_action['right'] == 1:
            actions.append(7)

        if len(actions) > 0:
            action = random.choice(actions)
        else:
            action = 4
            
        return action

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

def load_demo_buffer(env_name, max_items):
    env_wrapper = MineCraftWrapper(None)
    demo_buffer = ReplayBuffer(arglist.replay_buffer_len)
    data = minerl.data.make(environment=env_name, data_dir="./res")

    print("#############################################")
    print("Loading demonstrations")
    print("#############################################")
    items = 0
    for current_state, action, reward, next_state, done in data.batch_iter(batch_size=1, num_epochs=1, seq_len=500):
        for step in range(len(reward)):
            minerl_obs = {
                'pov': current_state['pov'][0][step],
                'compassAngle': current_state['compassAngle'][0][step]
            }            
            obs = env_wrapper.minerl_obs_to_obs(minerl_obs)
            minerl_new_obs = {
                'pov': next_state['pov'][0][step],
                'compassAngle': next_state['compassAngle'][0][step]
            }
            new_obs = env_wrapper.minerl_obs_to_obs(minerl_new_obs)
            minerl_action = {
                'attack': action['attack'][0][step],
                'back': action['back'][0][step],
                'camera': action['camera'][0][step],
                'forward': action['forward'][0][step],
                'jump': action['jump'][0][step],
                'left': action['left'][0][step],
                'right': action['right'][0][step]
            }
            action = env_wrapper.minerl_action_to_action(minerl_action)
            demo_buffer.add(obs, action, reward[0][step], new_obs, float(done[0][step]))

        items += 1
        if items >= max_items:
            break

    print("#############################################")
    print("Finished loading demonstrations")
    print("#############################################")
    
    return demo_buffer

def train_policy(arglist):
    with U.single_threaded_session():
        # Create the environment
        if arglist.use_dense_rewards:
            print("Will use env MineRLNavigateDense-v0")
            env = gym.make("MineRLNavigateDense-v0")    
            env_name = "MineRLNavigateDense-v0"     
        else:
            print("Will use env MineRLNavigate-v0")
            env = gym.make('MineRLNavigate-v0')
            env_name = "MineRLNavigate-v0"   

        if arglist.force_forward:
            env = MineCraftWrapperSimplified(env)
        else:
            env = MineCraftWrapper(env)

        env = FrameStack(env, 4)

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=build_q_func('conv_only', dueling=True),
            num_actions=env.action_space.n,
            gamma=0.9,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        # Create the replay buffer(s) (TODO: Use prioritized replay buffer)
        if arglist.use_demonstrations:
            replay_buffer = ReplayBuffer(int(arglist.replay_buffer_len/2))
            demo_buffer = load_demo_buffer(env_name, int(arglist.replay_buffer_len/2))
        else:
            replay_buffer = ReplayBuffer(arglist.replay_buffer_len)

        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=arglist.num_exploration_steps*arglist.num_episodes*arglist.max_episode_steps, initial_p=1.0, final_p=arglist.final_epsilon)

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
            print("Episode: ", str(episode))
            done = False
            episode_steps = 0
            while not done:
                
                # Take action and update exploration to the newest value
                action = act(obs[None], update_eps=exploration.value(n_steps))[0]
                new_obs, rew, done, _ = env.step(action)
                n_steps += 1
                episode_steps += 1

                # Break episode
                if episode_steps > arglist.max_episode_steps:
                    done =True

                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs              

                # Store rewards
                episode_rewards[-1] += rew
                if done:
                    obs = env.reset()
                    episode_rewards.append(0)
                    n_episodes += 1
          
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if (n_steps > arglist.learning_starts_at_steps) and (n_steps % 4 == 0):
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                if arglist.use_demonstrations:
                    if (n_steps < arglist.learning_starts_at_steps) and (n_steps % 4 == 0):
                        obses_t, actions, rewards, obses_tp1, dones = demo_buffer.sample(32)
                        train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    if (n_steps > arglist.learning_starts_at_steps) and (n_steps % 4 == 0):
                        obses_t, actions, rewards, obses_tp1, dones = demo_buffer.sample(32)
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

                #TODO: Save checkpoints
                if n_steps % arglist.checkpoint_rate == 0:
                    checkpoint_path = "./checkpoints/minerl_" + str(episode) + "_" + str(date.today()) + "_" + str(time.time()) + ".pkl"
                    save_variables(checkpoint_path)
                    print("%s,%s,%s,%s" % (n_steps,episode,round(np.mean(episode_rewards[-101:-1]), 1),int(100 * exploration.value(n_steps))), file=log_file)
        log_file.close()                 

def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN policy.')
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes to use for training") #default=100, help="number of episodes to use for training")
    parser.add_argument("--checkpoint-rate", type=int, default=10000, help="number of steps between checkpoints")
    parser.add_argument("--replay-buffer-len", type=int, default=100000, help="length of replay buffer") #default=30000, help="length of replay buffer")
    parser.add_argument("--num-exploration-steps", type=int, default=0.4, help="fraction of time steps to use for exploration") #default=0.2, help="fraction of time steps to use for exploration")
    parser.add_argument("--learning-starts-at-steps", type=int, default=10000, help="number of time steps before learning starts")
    parser.add_argument("--max-episode-steps", type=int, default=500, help="max number of time steps in an episode")
    parser.add_argument("--target-net-update-freq", type=int, default=1000, help="update frequency of the target network")
    parser.add_argument("--final-epsilon", type=float, default=0.02, help="final epsilon")
    parser.add_argument("--use-demonstrations", action="store_true", default=False)
    parser.add_argument("--use-dense-rewards", action="store_true", default=False)
    parser.add_argument("--force-forward", action="store_true", default=False)
    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    train_policy(arglist)
