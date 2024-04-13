import sys

from torch.distributions.normal import Normal
from policy_network import PolicyNetworkDiscrete
from torch.optim import AdamW


import numpy as np
import torch


class ReinforceAgent:

    def __init__(self, policy):
        self.learning_rate = 10e-4
        self.gamma = 0.9
        self.eps = 1e-6

        self.probabilities_for_action = []
        self.rewards_for_action = []

        self.policy_network = policy
        self.optimizer = AdamW(self.policy_network.parameters(), lr=self.learning_rate)

    def sample_action_discrete(self, state: np.ndarray) -> int:
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.policy_network(state)

        distribution = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distribution.sample()
        probability = distribution.log_prob(action)

        chosen_action = int(np.argmax(action.numpy()))
        self.probabilities_for_action.append(probability)

        # print(f'Chosen action: {chosen_action}')
        # print(f'Distribution: {action}')

        return chosen_action

    def update_policy_network(self):

        running_g = 0
        gs = []

        for R in self.rewards_for_action[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        for log_prob, delta in zip(self.probabilities_for_action, deltas):
            loss += log_prob.mean() * delta * (-1)

        """
        for param in self.policy_network.shared_net.parameters():
            if len(param.data.numpy().shape) == 1 and param.data.numpy().shape[0] == 8:
                print(param.data.numpy())
        """

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        """
        for param in self.policy_network.shared_net.parameters():
            if len(param.data.numpy().shape) == 1 and param.data.numpy().shape[0] == 8:
                print(param.data.numpy())

        print('----------------------------------------------------------------------')
        """

        self.probabilities_for_action = []
        self.rewards_for_action = []


