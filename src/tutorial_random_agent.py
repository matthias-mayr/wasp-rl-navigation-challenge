# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:46:14 2020

@author: hecro15
"""

import gym
import minerl

def main():
    env = gym.make('MineRLNavigateDense-v0')
    _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()

if __name__ == '__main__':
    main()