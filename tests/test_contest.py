import unittest
from multiprocessing import Process, Queue
from typing import List

from agent.batch_inference import SharedData, contestInference
from agent.mcts import MCTSPlayer
from agent.network import PolicyValueNet
from train_utils.game import contest

from config import MCTS_CONFIG


class TestContest(unittest.TestCase):
    def testContest(self):
        n_process = 2
        net0 = PolicyValueNet()
        # net0.load("xxx")
        # net1 = PolicyValueNet()

        with SharedData(n_process) as shared_data:
            done_queue = Queue()
            shared_data.reset()
            processes: List[Process] = []

            # >>>>> case1: net0 vs net1
            # for i in range(n_process):
            #     processes.append(
            #         Process(target=contest,
            #                 args=(i, i, shared_data, done_queue)))
            #     processes[-1].start()
            # contestInference(
            #     shared_data, (net0, net1),
            #     (MCTS_CONFIG.n_search, ) * 2)

            # >>>>> case2: net0 vs pure MCTS
            players = [None, MCTSPlayer(n_search=1000)]
            for i in range(n_process):
                processes.append(
                    Process(target=contest,
                            args=(i, i, shared_data,
                                  done_queue, players)))
                processes[-1].start()
            contestInference(
                shared_data, (net0, None),
                (MCTS_CONFIG.n_search, 0))

            for _ in range(n_process):
                done_queue.get()
            for proc in processes:
                proc.join()
