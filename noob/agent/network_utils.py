import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MDP_CONFIG, NETWORK_CONFIG
from env.simulator import Simulator
from icecream import ic


class ObsEncoder():
    @staticmethod
    def encode(env: Simulator) -> np.ndarray:
        """[summary]
        features selection

        NOTE: e.g. when NETWORK_CONFIG.periods_num = 5,
            return matrix with shape: (5 * 2 + 1, BOARD_SIZE, BOARD_SIZE),
            1. current position of black and the previous 4 time periods
                1 if black stone here, 0 if back stone not here
            2. current position of white and the previous 4 time periods
            3. all 0/1 matrix (indicate who is to play, same as env.turn)

        """
        periods_num = NETWORK_CONFIG.periods_num
        features = np.zeros(
            (periods_num * 2 + 1, *(MDP_CONFIG.board_size, ) * 2))

        env = copy.deepcopy(env)

        # who is to play
        if env.turn == 1:
            features[-1, ...] = np.ones(
                (MDP_CONFIG.board_size, ) * 2)

        for i in range(periods_num):
            # black
            features[i, ...] = (env.board == 0).astype(int)
            # white
            features[i + periods_num, ...] = (env.board == 1).astype(int)

            # backtrack
            for _ in range(2):
                if env.actions:
                    index = env.actions[-1]
                    env.backtrack(index)

        return features


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, padding=1, bias=False)


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
