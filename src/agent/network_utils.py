import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from env.gobang_env import GobangEnv

from config import ENV_CONFIG, NETWORK_CONFIG


class ObsEncoder():
    @staticmethod
    def encode(env: GobangEnv) -> np.ndarray:
        """[summary]
        feature selection

        NOTE: e.g. when NETWORK_CONFIG.n_periods = 5,
            return matrix with shape: (5 * 2 + 1, BOARD_SIZE, BOARD_SIZE),
            1. current positions (0-1 matrix) of black stones and the previous 4 time periods
            2. current positions of white stones and the previous 4 time periods
            3. all 0/1 matrix (indicate who is to play, == env.turn)
        """
        # deepcopy
        env = copy.deepcopy(env)

        n_periods = NETWORK_CONFIG.n_periods
        board_size = (ENV_CONFIG.board_size, ) * 2
        features = np.zeros(
            (n_periods * 2 + 1, *board_size), dtype=np.float32)

        # who is to play
        if env.turn == 1:
            features[-1, ...] = np.ones(board_size)

        for i in range(n_periods):
            # black stone
            features[i, ...] = (env.board == 0)
            # white stone
            features[i + n_periods, ...] = (env.board == 1)

            # backtrack
            for _ in range(2):
                if env.action_log:
                    env.backtrack()

        return features


def conv3x3(in_channels, out_channels):
    # NOTE: BN subtracts mean, which cancels the effect of bias
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, padding=1, bias=False
    )


class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.net = nn.Sequential(
            conv3x3(n_channels, n_channels),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            conv3x3(n_channels, n_channels),
            nn.BatchNorm2d(n_channels)
        )

    def forward(self, x):
        y = self.net(x)
        return F.relu(x + y)
