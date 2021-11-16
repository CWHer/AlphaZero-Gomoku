import pickle
from collections import deque, namedtuple

import numpy as np
import torch
from config import MDP_CONFIG, TRAIN_CONFIG
from torch.utils.data import DataLoader, TensorDataset
from utils import plotHeatMap, plotSparseMatrix


class ReplayBuffer():
    Data = namedtuple("data", "state mcts_prob value")

    def __init__(self) -> None:
        self.buffer = deque(maxlen=TRAIN_CONFIG.replay_size)

    def save(self, version="w"):
        dataset_dir = TRAIN_CONFIG.dataset_dir

        import os
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        print("save replay buffer {}".format(version))
        with open(dataset_dir +
                  f"/data_{version}.pkl", "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, data_dir):
        print("load replay buffer {}".format(data_dir))
        with open(data_dir, "rb") as f:
            self.buffer = pickle.load(f)

    def enough(self):
        """[summary]
        whether data is enough to start training
        """
        return len(self.buffer) > TRAIN_CONFIG.train_threshold

    def __enhanceData(self, states, mcts_probs, values):
        """[summary]
        enhance data by rotating and flipping

        """
        # TODO: debug
        board_size, data = MDP_CONFIG.board_size, []
        for state, mcts_prob, value in zip(states, mcts_probs, values):
            for i in range(4):
                # debug
                plotHeatMap(mcts_prob, "none")
                plotSparseMatrix(state[0], "none")

                # rotate
                new_state = np.rot90(state, i, axes=(1, 2))
                new_mcts_prob = np.rot90(
                    mcts_prob.reshape((board_size, ) * 2), i)
                data.append((new_state, new_mcts_prob.flatten(), value))

                # debug
                plotHeatMap(new_mcts_prob, "none")
                plotSparseMatrix(new_state[0], "board")

                # flip
                new_state = np.array([np.fliplr(s) for s in new_state])
                new_mcts_prob = np.fliplr(new_mcts_prob)
                data.append((new_state, new_mcts_prob.flatten(), value))

                # debug
                plotHeatMap(new_mcts_prob, "none")
                plotSparseMatrix(new_state[0], "board")

        return data

    def add(self, states, mcts_probs, values):
        """[summary]
        """
        self.buffer.extend(
            self.__enhanceData(states, mcts_probs, values))

    def trainIter(self):
        """[summary]
        generate dataset iterator for training
        """
        states, mcts_probs, values = map(torch.tensor, zip(*self.buffer))
        data_set = TensorDataset(states, mcts_probs, values)
        train_iter = DataLoader(
            data_set, TRAIN_CONFIG.batch_size, shuffle=True)
        return train_iter
