# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:46:14 2020

@author: hecro15
"""

import gym
import minerl

def preprocess_image_frame():
    """Preprocess image frame given from the environment."""
    raise NotImplementedError

def build_q_network():
    """Build a neural network for the q function."""
    raise NotImplementedError

class MineCraftWrapper:
    """Wrap the action and observation spaces of the MineCraft environment."""
    def __init__(self, minecraft_env):
        self.minecraft_env = minecraft_env
        
class Agent:
    """A DQN agent."""
    def __init__(self):
        raise NotImplementedError
    
    def act(self, obs):
        """Get action for observation."""
        raise NotImplementedError
    
    def add_experience(self, obs, act, rew, next_obs):
        """Add an experience to the replay buffer."""
        raise NotImplementedError
    
    def learn(self):
        """Update the agent's policy."""
        raise NotImplementedError
    
    def update_target_network(self):
        """Update the target network of the agent."""
        raise NotImplementedError
        
    def save_checkpoint(self, checkpoint_name):
        """Save a checkpoint with the current state of the agent."""
        raise NotImplementedError
        
    def load_checkpoint(self, checkpoint_name):
        """Load a checkpoint with the a state of the agent."""
        raise NotImplementedError

def main():
    env = gym.make('MineRLNavigate-v0')
    env = MineCraftWrapper(env)

if __name__ == '__main__':
    main()
