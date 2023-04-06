import pickle
import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import plotHeatMap, plotSparseMatrix, printInfo

from config import DATA_CONFIG, ENV_CONFIG


class ReplayBuffer():
    """[summary]
    NOTE: data = (state, mcts_prob, mcts_val)
    """

    def __init__(self) -> None:
        self.buffer = deque(
            maxlen=DATA_CONFIG.replay_size)

    def __len__(self):
        return len(self.buffer)

    def save(self, version="default"):
        dataset_dir = DATA_CONFIG.dataset_dir

        import os
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        printInfo(f"save replay buffer version({version})")
        with open(dataset_dir +
                  f"/data_{version}.pkl", "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, data_path):
        printInfo(f"load replay buffer {data_path}")
        with open(data_path, "rb") as f:
            self.buffer = pickle.load(f)

    def enough(self):
        return len(self) > DATA_CONFIG.train_threshold

    def __augmentData(
            self, states, mcts_probs, mcts_vals):
        """[summary]
        augment data by rotating and flipping
        """
        data_buffer = []
        board_size = (ENV_CONFIG.board_size, ) * 2
        for state, mcts_prob, mcts_val in \
                zip(states, mcts_probs, mcts_vals):

            for i in range(4):
                # rotate
                new_state = np.rot90(state, i, axes=(1, 2))
                new_mcts_prob = np.rot90(
                    mcts_prob.reshape(board_size), i)
                data_buffer.append(
                    (new_state, new_mcts_prob.flatten(), mcts_val))
                # plotHeatMap(
                #     new_mcts_prob, name=f"prob_rotate{i}")
                # plotSparseMatrix(
                #     new_state[0], name=f"state_rorate{i}")

                if not DATA_CONFIG.augment_data:
                    break

                # flip
                new_state = np.array(
                    [np.fliplr(s) for s in new_state])
                new_mcts_prob = np.fliplr(new_mcts_prob)
                data_buffer.append(
                    (new_state, new_mcts_prob.flatten(), mcts_val))
                # plotHeatMap(
                #     new_mcts_prob, name=f"prob_flip{i}")
                # plotSparseMatrix(
                #     new_state[0], name=f"state_flip{i}")

        return data_buffer

    def add(self, states, mcts_probs, mcts_vals):
        self.buffer.extend(
            self.__augmentData(states, mcts_probs, mcts_vals))

    def sample(self) -> DataLoader:
        selected_data = random.sample(
            self.buffer, DATA_CONFIG.sample_size)
        states, mcts_probs, mcts_vals = map(
            lambda x: torch.from_numpy(np.array(x)),
            zip(*selected_data)
        )
        train_iter = DataLoader(
            TensorDataset(states, mcts_probs, mcts_vals),
            DATA_CONFIG.batch_size, shuffle=True
        )
        return train_iter
