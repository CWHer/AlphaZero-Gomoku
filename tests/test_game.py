from multiprocessing import Process, Queue
from typing import List

from icecream import ic

from agent.batch_inference import SharedData, batchInference
from agent.network import PolicyValueNet
from train_utils.game import selfPlay
from train_utils.replay_buffer import ReplayBuffer


buffer = ReplayBuffer()
n_process, n_epoch = 10, 2
net = PolicyValueNet()
net.setDevice()

with SharedData(n_process) as shared_data:
    done_queue = Queue()
    for _ in range(n_epoch):
        shared_data.reset()
        processes: List[Process] = []

        for i in range(n_process):
            processes.append(
                Process(target=selfPlay,
                        args=(i, i, shared_data, done_queue)))
            processes[-1].start()
        batchInference(shared_data, net)

        for _ in range(n_process):
            buffer.add(*done_queue.get())
        for proc in processes:
            proc.join()

buffer.save()
ic(len(buffer))
for i in range(100):
    train_iter = buffer.sample()
    for data_batch in train_iter:
        loss, acc = net.trainStep(data_batch)
        ic(loss, acc)
net.save()
