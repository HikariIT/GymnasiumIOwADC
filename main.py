from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from torch.distributions.normal import Normal
from models import ModelType

from policy_network import PolicyNetworkDiscrete, PolicyNetworkContinuous

import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch
import colorsys

from reinforce import ReinforceAgent

def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / float(num_colors)
        rgb = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
        colors.append('#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2]))
    return colors

def calculate_frozen_lake():
    frozen_lake_env = gym.make(
        ModelType.FROZEN_LAKE,
        is_slippery=False,
        desc=None,
        map_name='4x4'
    )

    wrapped_env = gym.wrappers.RecordEpisodeStatistics(frozen_lake_env, 200)
    total_no_episodes = 10000

    observation_space_dim = frozen_lake_env.observation_space.n
    action_space_dim = frozen_lake_env.action_space.n

    print(observation_space_dim, action_space_dim)

    avg_rewards_for_seeds = {

    }

    rewards_per_episode_per_seed = {}

    for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        agent = ReinforceAgent(PolicyNetworkDiscrete(observation_space_dim, action_space_dim))
        reward_over_episodes = []

        for episode in range(total_no_episodes):
            position, _ = wrapped_env.reset(seed=seed)

            observation = np.zeros((observation_space_dim,))
            observation[position] = 1

            done = False
            while not done:
                action = agent.sample_action_discrete(observation)
                position, reward, terminated, truncated, info = wrapped_env.step(action)

                observation = np.zeros((observation_space_dim,))
                observation[position] = 1

                agent.rewards_for_action.append(reward)
                done = terminated or truncated

            agent.update_policy_network()

            if episode % 1000 == 0:
                avg_reward = np.mean(wrapped_env.return_queue)
                print(f'Episode: {episode}, Average Reward: {avg_reward:.5f}')
                reward_over_episodes.append((episode, avg_reward))

        rewards_per_episode_per_seed[seed] = reward_over_episodes.copy()
        avg_rewards_for_seeds[seed] = np.mean(wrapped_env.return_queue)

    for seed, avg_reward in avg_rewards_for_seeds.items():
        print(f'Seed: {seed}, Average Reward after {total_no_episodes} episodes: {avg_reward:.5f}')

    seeds = list(rewards_per_episode_per_seed.keys())
    num_seeds = len(seeds)
    colors = generate_colors(num_seeds)  # colormap based on the number of seeds

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, seed in enumerate(seeds):
        episodes, avg_rewards = zip(*rewards_per_episode_per_seed[seed])
        color = colors[i]  # get color based on index
        plt.scatter(episodes, avg_rewards, color=color, marker='o', label=seed)

    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Episode for Different Seeds')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file (e.g., PNG format)
    plt.savefig('reward_plot.png')

    # Display the plot
    plt.show()



def calculate_car_racing():
    env = gym.make("CarRacing-v2", domain_randomize = True, continuous = True)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    total_num_episodes = int(5e3)  # Total number of episodes
    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.observation_space.shape[0]
    print(obs_space_dims)
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space.shape[0]
    print(action_space_dims)
    rewards_over_seeds = []

    for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        agent = ReinforceAgent(PolicyNetworkContinuous(obs_space_dims, action_space_dims))
        reward_over_episodes = []

        for episode in range(total_num_episodes):
            # gymnasium v26 requires users to set seed while resetting the environment
            obs, info = wrapped_env.reset(seed=seed)

            done = False
            while not done:
                action = agent.sample_action(obs)

                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards_for_action.append(reward)

                done = terminated or truncated

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            agent.update_policy_network()

            if episode % 1000 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print("Episode:", episode, "Average Reward:", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)

        rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
        df1 = pd.DataFrame(rewards_to_plot).melt()
        df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
        sns.set(style="darkgrid", context="talk", palette="rainbow")
        sns.lineplot(x="episodes", y="reward", data=df1).set(
            title="REINFORCE for Car-Racingv2"
        )
        plt.show()

def main():
    calculate_car_racing()



if __name__ == '__main__':
    main()