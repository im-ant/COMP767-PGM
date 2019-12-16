# =============================================================================
# The DQN network architecture
#
# Very much inspired by / adopted from:
# https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py
#
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_output_length(input_length: int, kernel_size: int,
                          stride: int) -> int:
    """
    Compute the output (side) length of a conv2d operation

    :param input_length: length of the input 2d matrix along this dimension
    :param kernel_size: size of the kernel along this dimension
    :param stride: stride of the kernel along this dimension
    :return: int denoting the length of output
    """
    out_length = input_length - (kernel_size - 1) - 1
    out_length = out_length / stride + 1
    return int(out_length)


class nature_dqn_network(nn.Module):
    """
    The Mnih 2015 Nature DQN network, as described in
    https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning
    """

    def __init__(self, num_actions: int, num_channels: int,
                 input_shape: Tuple = (84, 84)):
        """

        :param num_actions: number of allowable actions
        :param num_channels: number of input image channels (i.e. number of
                stacked frames)
        """
        super(nature_dqn_network, self).__init__()

        self.num_actions = num_actions
        self.num_channels = num_channels
        self.input_shape = input_shape

        # Initialize conv layers
        self.conv1 = nn.Conv2d(self.num_channels, 32,
                               kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)

        # Compute num of units: https://pytorch.org/docs/stable/nn.html#conv2d
        # also github.com/transedward/pytorch-dqn/blob/master/dqn_model.py
        # NOTE: assume square kernel
        side_length = compute_output_length(input_shape[0], 8, 4)
        side_length = compute_output_length(side_length, 4, 2)
        side_length = compute_output_length(side_length, 3, 1)

        # Initialize fully connected layers; num units computed from:
        self.fc1 = nn.Linear(64 * side_length * side_length, 512)
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h = h.view(h.shape[0], -1)  # flatten convolution
        h = F.relu(self.fc1(h))
        return self.fc2(h)


if __name__ == "__main__":
    # for testing run this directly
    print('testing')
    test_dqn = nature_dqn_network(8, 64)
    print(test_dqn)


