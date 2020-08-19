# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:46:14 2020

@author: hecro15
"""

import gym
import minerl

def main():
    env = gym.make('MineRLNavigateDense-v0')
    obs  = env.reset()
    done = False
    net_reward = 0
    while not done:
        action = env.action_space.noop()
    
        action['camera'] = [0, 0.03*obs["compassAngle"]]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1
        
        obs, rew, done, _ = env.step(action)
        env.render()
        
        net_reward += rew
        print("Total reward: ", net_reward)

if __name__ == '__main__':
    main()