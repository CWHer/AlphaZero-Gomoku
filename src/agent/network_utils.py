import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from config import ENV_CONFIG, NETWORK_CONFIG
from env.simulator import Simulator


class ObsEncoder():
    @staticmethod
    def encode(env: Simulator) -> np.ndarray:
        """[summary]
        feature selection

        NOTE: e.g. when NETWORK_CONFIG.periods_num = 5,
            return matrix with shape: (5 * 2 + 1, BOARD_SIZE, BOARD_SIZE),
            1. current positions (0-1 matrix) of black stones and the previous 4 time periods
            2. current positions of white stones and the previous 4 time periods
            3. all 0/1 matrix (indicate who is to play, == env.turn)
        """
        # deepcopy
        env = copy.deepcopy(env)

        periods_num = NETWORK_CONFIG.periods_num
        board_size = (ENV_CONFIG.board_size, ) * 2
        features = np.zeros(
            (periods_num * 2 + 1, *board_size), dtype=np.float32)

        # who is to play
        if env.turn == 1:
            features[-1, ...] = np.ones(board_size)

        for i in range(periods_num):
            # black stone
            features[i, ...] = (env.board == 0)
            # white stone
            features[i + periods_num, ...] = (env.board == 1)

            # backtrack
            for _ in range(2):
                if env.action_log:
                    env.backtrack()

        return features


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1)


class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.net = nn.Sequential(
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self, x):
        y = self.net(x)
        return F.relu(x + y)
