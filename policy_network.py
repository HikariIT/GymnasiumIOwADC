import torch.nn as nn
import torch


class PolicyNetworkDiscrete(nn.Module):

    def __init__(self, observation_space_dim: int, action_space_dim: int):
        super().__init__()

        shared_net_layers = []
        neurons_per_layer = [observation_space_dim, 32, 16]

        for i in range(1, len(neurons_per_layer)):
            shared_net_layers.extend([
                nn.Linear(neurons_per_layer[i - 1], neurons_per_layer[i]),
                nn.ReLU()
            ])

        self.shared_net = nn.Sequential(*shared_net_layers)
        self.policy_mean_net = nn.Sequential(nn.Linear(neurons_per_layer[-1], action_space_dim), nn.Sigmoid())
        self.policy_stddev_net = nn.Sequential(nn.Linear(neurons_per_layer[-1], action_space_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(shared_features)))

        return action_means, action_stddevs
    

class PolicyNetworkContinuous(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, observation_space_dim: int, action_space_dim: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        shared_net_layers = []
        neurons_per_layer = [observation_space_dim, 32, 16]

        for i in range(1, len(neurons_per_layer)):
            shared_net_layers.extend([
                nn.Linear(neurons_per_layer[i - 1], neurons_per_layer[i]),
                nn.Tanh()
            ])

        self.shared_net = nn.Sequential(*shared_net_layers)
        self.policy_mean_net = nn.Sequential(nn.Linear(neurons_per_layer[-1], action_space_dim), nn.Tanh())
        self.policy_stddev_net = nn.Sequential(nn.Linear(neurons_per_layer[-1], action_space_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs
