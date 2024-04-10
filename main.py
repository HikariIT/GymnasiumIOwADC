from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from torch.distributions.normal import Normal
from models import ModelType

import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch

from reinforce import ReinforceAgent


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

    for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        agent = ReinforceAgent(observation_space_dim, action_space_dim)
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

        avg_rewards_for_seeds[seed] = np.mean(wrapped_env.return_queue)

    for seed, avg_reward in avg_rewards_for_seeds.items():
        print(f'Seed: {seed}, Average Reward after {total_no_episodes} episodes: {avg_reward:.5f}')

def main():
    calculate_frozen_lake()


if __name__ == '__main__':
    main()