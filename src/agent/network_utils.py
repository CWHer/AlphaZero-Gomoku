import copy

import numpy as np
import torch.nn as nn
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = False
        if in_channels != out_channels:
            self.downsample = True
            self.downsample_conv = conv3x3(in_channels, out_channels)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample_conv(residual)
            residual = self.downsample_bn(residual)

        out += residual
        out = self.relu(out)
        return out
