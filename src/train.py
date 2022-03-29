import copy
import random
from functools import partial

from tensorboardX import SummaryWriter
from torch import multiprocessing
from torch.multiprocessing import Pool
from tqdm import tqdm

from agent.network import PolicyValueNet
from config import DATA_CONFIG, TRAIN_CONFIG
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

        games = [
            (self.net, self.seed())
            for _ in range(n_game)
        ]
        with tqdm(total=n_game) as pbar:
            with Pool(TRAIN_CONFIG.n_process) as pool:
                results = pool.starmap(selfPlay, games)
                for result in results:
                    self.buffer.add(*result)
                    pbar.update()

    @timeLog
    def __evaluate(self):
        printInfo("evaluate model")

        logs = {0: 0, 1: 0, -1: 0}
        n_contest = TRAIN_CONFIG.n_contest
        with tqdm(total=n_contest) as pbar:
            games = [
                (self.net, self.best_net, self.seed())
                for _ in range(n_contest // 2)
            ]
            with Pool(TRAIN_CONFIG.n_process) as pool:
                results = pool.starmap(contest, games)
                for winner in results:
                    logs[winner] += 1
                    pbar.update()

            games = [
                (self.best_net, self.net, self.seed())
                for _ in range(n_contest // 2)
            ]
            with Pool(TRAIN_CONFIG.n_process) as pool:
                results = pool.starmap(contest, games)
                for winner in results:
                    if winner != -1:
                        winner ^= 1
                    logs[winner] += 1
                    pbar.update()

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
    multiprocessing.set_start_method("spawn")

    trainer = Trainer(loc="cuda:0")
    trainer.run()
