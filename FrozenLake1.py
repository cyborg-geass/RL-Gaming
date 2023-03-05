import numpy as np
import seaborn as sns
import gymnasium as gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v1", render_mode = 'ansi')
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

num_episodes = 10000
max_steps_per_episodes = 100

learning_rate = 0.01
discount_rate = 0.99
exploration_rate =1
max_exploration_rate = 1
min_exoloration_rate = 0.01
exploration_decay_rate = 0.001

reward_all_episodes = []

for episode in range(num_episodes):

    state = env.reset()
    rewards_per_episodes = 0
    done = False

    for step in range(max_steps_per_episodes):
        random_rate = np.random.uniform(0,1)
        if random_rate>exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        new_state, reward, done, truncated, info = env.step(action)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
        learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        rewards_per_episodes+=reward
        state = new_state
        if done==True:
           break
    
    exploration_rate = exploration_rate + (max_exploration_rate-min_exoloration_rate)*np.exp(exploration_decay_rate*episode)
    reward_all_episodes.append(rewards_per_episodes)

print("********Average reward per thousand episodes********\n")
rewards_per_thousand_episodes = np.split(np.array(reward_all_episodes),num_episodes/1000)
count = 1000
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
print("\n\n********Q-table********\n")
print(q_table)
