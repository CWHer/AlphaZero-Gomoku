from multiprocessing import Condition, Value
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple

import numpy as np
from config import ENV_CONFIG, NETWORK_CONFIG
from tqdm import tqdm

from agent.network import PolicyValueNet


class SharedData():
    """[summary]
    NOTE: can only used with context manager
    """

    def __init__(self, n_proc) -> None:
        self.n_proc = n_proc
        self.pbar = tqdm(total=n_proc)

        self.player_cv = Condition()
        self.inference_cv = Condition()

        self.n_waiting = Value("i", 0)
        self.n_finished = Value("i", 0)

    def __initSharedMemory(self, n_proc):
        def f32Bytes(x): return x * 4

        # features
        board_size = (ENV_CONFIG.board_size, ) * 2
        n_board = ENV_CONFIG.board_size ** 2
        n_periods = NETWORK_CONFIG.n_periods * 2 + 1

        self.features_shm = SharedMemory(
            create=True, size=f32Bytes(n_proc * n_periods * n_board))
        self.features = np.ndarray(
            (n_proc, n_periods, *board_size),
            dtype=np.float32, buffer=self.features_shm.buf
        )
        # policy
        self.probs_shm = SharedMemory(
            create=True, size=f32Bytes(n_proc * n_board))
        self.probs = np.ndarray(
            (n_proc, n_board), dtype=np.float32, buffer=self.probs_shm.buf)
        # value
        self.values_shm = SharedMemory(
            create=True, size=f32Bytes(n_proc))
        self.values = np.ndarray(
            n_proc, dtype=np.float32, buffer=self.values_shm.buf)

    def __delSharedMemory(self):
        self.features_shm.close()
        self.features_shm.unlink()

        self.probs_shm.close()
        self.probs_shm.unlink()

        self.values_shm.close()
        self.values_shm.unlink()

    def __enter__(self):
        self.__initSharedMemory(self.n_proc)
        return self

    def __exit__(self, *_):
        self.__delSharedMemory()

    def reset(self):
        self.n_finished.value = 0
        self.pbar.reset()

    # >>>>>> player APIs
    def put(self, k, features: Optional[np.ndarray]):
        if not features is None:
            self.features[k, ...] = features

        with self.player_cv:
            with self.n_waiting.get_lock():
                self.n_waiting.value += 1
            with self.inference_cv:
                self.inference_cv.notify()

            # NOTE: this prevents the
            #   waiting++ --> notify_all --> wait deadlock
            self.player_cv.wait()

    def get(self, k):
        return self.probs[k], self.values[k]

    def finish(self):
        with self.n_finished.get_lock():
            self.n_finished.value += 1
            self.pbar.update(self.n_finished.value)
        with self.inference_cv:
            self.inference_cv.notify()
    # <<<<<< player APIs

    # >>>>>> inference APIs
    def isDone(self):
        with self.n_finished.get_lock():
            return self.n_finished.value == self.n_proc

    def isReady(self):
        with self.n_waiting.get_lock() \
                and self.n_finished.get_lock():
            return self.n_finished.value + \
                self.n_waiting.value == self.n_proc

    def wait(self):
        with self.inference_cv:
            self.inference_cv.wait_for(self.isReady)

        self.n_waiting.value = 0

    def notify(self):
        with self.player_cv:
            self.player_cv.notify_all()
    # <<<<<< inference APIs


def batchInference(shared_data: SharedData,
                   net: PolicyValueNet):

    while not shared_data.isDone():
        # players put features
        shared_data.wait()

        # batch inference
        shared_data.probs[...], shared_data.values[...] = \
            net.predict(shared_data.features)

        shared_data.notify()
        # players get results


def contestInference(shared_data: SharedData,
                     net: Tuple, n_search: Tuple):
    """[summary]
    NOTE: all MCTS player must have the same n_search
    """
    index = 0
    while not shared_data.isDone():
        for _ in range(n_search[index]):
            # players put features
            shared_data.wait()

            # batch inference
            shared_data.probs[...], shared_data.values[...] = \
                net[index].predict(shared_data.features)

            shared_data.notify()
            # players get results

        index ^= 1
