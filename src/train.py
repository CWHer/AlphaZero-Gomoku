import copy
import random
from functools import partial
from multiprocessing import Process, Queue
from typing import List

from tensorboardX import SummaryWriter
from tqdm import tqdm

from agent.batch_inference import SharedData, batchInference, contestInference
from agent.network import PolicyValueNet
from config import DATA_CONFIG, MCTS_CONFIG, TRAIN_CONFIG
from train_utils.game import contest, selfPlay
from train_utils.replay_buffer import ReplayBuffer
from utils import printInfo, printWarn, timeLog


class Trainer():
    def __init__(self, loc="cpu") -> None:
        self.net = PolicyValueNet()
        self.net.setDevice(loc)
        self.buffer = ReplayBuffer()

        self.n_best = 0
        self.best_net = copy.deepcopy(self.net)

        self.seed = partial(
            random.randint, a=0, b=20000905)
        self.writer = SummaryWriter(TRAIN_CONFIG.log_dir)

    @timeLog
    def __collectData(self):
        printInfo("collect data")

        n_game = TRAIN_CONFIG.n_game
        n_process = TRAIN_CONFIG.n_process
        with SharedData(n_process) as shared_data:
            done_queue = Queue()
            # each epoch contrains n_process
            for _ in range(n_game // n_process):
                shared_data.reset()
                processes: List[Process] = []

                for i in range(n_process):
                    processes.append(
                        Process(target=selfPlay,
                                args=(self.seed(), i,
                                      shared_data, done_queue)))
                    processes[-1].start()
                # batch inference
                batchInference(shared_data, self.net)

                for _ in range(n_process):
                    self.buffer.add(*done_queue.get())
                for proc in processes:
                    proc.join()

    @timeLog
    def __evaluate(self):
        printInfo("evaluate model")

        logs = {0: 0, 1: 0, -1: 0}
        n_contest = TRAIN_CONFIG.n_contest // 2
        n_process = TRAIN_CONFIG.n_process

        with SharedData(n_process) as shared_data:
            done_queue = Queue()
            # each epoch contrains n_process
            for _ in range(n_contest // n_process):
                shared_data.reset()
                processes: List[Process] = []

                for i in range(n_process):
                    processes.append(
                        Process(target=contest,
                                args=(self.seed(), i,
                                      shared_data, done_queue)))
                    processes[-1].start()
                # alternate inferene
                contestInference(
                    shared_data, (self.net, self.best_net),
                    (MCTS_CONFIG.n_search, ) * 2)

                for _ in range(n_process):
                    winner = done_queue.get()
                    logs[winner] += 1
                for proc in processes:
                    proc.join()

            # swap order
            for _ in range(n_contest // n_process):
                shared_data.reset()
                processes: List[Process] = []

                for i in range(n_process):
                    processes.append(
                        Process(target=contest,
                                args=(self.seed(), i,
                                      shared_data, done_queue)))
                    processes[-1].start()
                # alternate inferene
                contestInference(
                    shared_data, (self.best_net, self.net),
                    (MCTS_CONFIG.n_search, ) * 2)

                for _ in range(n_process):
                    winner = done_queue.get()
                    if winner != -1:
                        winner ^= 1
                    logs[winner] += 1
                for proc in processes:
                    proc.join()

        printInfo(
            f"win: {logs[0]}, lose: {logs[1]}, draw: {logs[-1]}")

        return (logs[0] + 0.5 * logs[-1]) / n_contest

    def __train(self, epoch):
        printInfo("train model")

        train_iter = self.buffer.sample()
        n_batch, mean_loss, mean_acc = 0, 0, 0
        with tqdm(total=len(train_iter)) as pbar:
            for data_batch in train_iter:
                n_batch += 1
                loss, acc = \
                    self.net.trainStep(data_batch)
                mean_loss += loss
                mean_acc += acc
                pbar.update()

        mean_loss /= n_batch
        mean_acc /= n_batch
        self.writer.add_scalar("loss", mean_loss, epoch)
        self.writer.add_scalar("accuracy", mean_acc, epoch)
        printInfo(
            "loss: {:>.4f}, accuracy: {:>.4f}".format(
                mean_loss, mean_acc)
        )

    def run(self):
        """[summary]
        NOTE: training pipeline contains:
             1. collect data    2. train model
             3. evaluate model  4. update best model
        """

        for i in range(TRAIN_CONFIG.epochs):
            printInfo(f"=====Epoch {i}=====")

            # >>>>> collect data
            self.__collectData()
            printInfo(f"buffer size {len(self.buffer)}")
            # save data
            if (i + 1) % DATA_CONFIG.save_freq == 0:
                self.buffer.save(version=f"epoch{i}")

            # >>>>> train model
            if self.buffer.enough():
                self.__train(epoch=i)

            # >>>>> evaluate model
            if (i + 1) % TRAIN_CONFIG.eval_freq == 0:
                if self.__evaluate() >= \
                        TRAIN_CONFIG.update_thr:
                    # update best model
                    self.n_best += 1
                    self.best_net = copy.deepcopy(self.net)
                    self.best_net.save(version=self.n_best)
                    printInfo(f"best model {self.n_best}")
                else:
                    printWarn(True, "reject new model")


if __name__ == "__main__":
    trainer = Trainer(loc="cuda:0")
    trainer.run()
