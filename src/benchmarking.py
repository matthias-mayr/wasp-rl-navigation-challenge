import gym
import minerl
import logging
import time
logging.basicConfig(level=logging.INFO)

env = gym.make('MineRLNavigate-v0')

obs  = env.reset()
done = False
net_reward = 0

num_episodes = 10

time_steps = 0
num_steps = 0
time_reset = 0
num_resets = 0

while num_resets < num_episodes:
    episode_steps = 0
    done = False
    while not done:
        # Keep the heuristics policy
        action = env.action_space.noop()
        action['camera'] = [0, 0.03*obs["compassAngle"]]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1

        start = time.time()
        obs, reward, done, info = env.step(
            action)
        end = time.time()
        time_steps += (end - start)
        num_steps += 1
        episode_steps += 1
        net_reward += reward
        if episode_steps > 500:
            done = True
    start = time.time()
    env.reset()
    end = time.time()
    time_reset += (end-start)
    num_resets += 1
    print("Finished run", num_resets, "of", num_episodes)

print("Benchmarking summary:")
print("Total time spent for steps: ", time_steps)
print("Time per step: ", time_steps/num_steps)
print("Total time to reset the environment: ", time_reset)
print("Time per environment reset: ", time_reset/num_resets)
