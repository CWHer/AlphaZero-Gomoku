from multiprocessing import Process, Queue
from typing import List

from icecream import ic

from agent.batch_inference import SharedData, contestInference
from agent.network import PolicyValueNet
from config import MCTS_CONFIG
from train_utils.game import contest


n_process = 4
net0 = PolicyValueNet()
# net0.load("xxx")
net1 = PolicyValueNet()

with SharedData(n_process) as shared_data:
    done_queue = Queue()
    shared_data.reset()
    processes: List[Process] = []

    for i in range(n_process):
        processes.append(
            Process(target=contest,
                    args=(i, i, shared_data, done_queue)))
        processes[-1].start()
    contestInference(
        shared_data, (net0, net1),
        (MCTS_CONFIG.n_search, ) * 2)

    for _ in range(n_process):
        ic(done_queue.get())
    for proc in processes:
        proc.join()
