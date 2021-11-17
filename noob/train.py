import copy

from icecream import ic
from tqdm import tqdm

from agent.network import PolicyValueNet
from config import TRAIN_CONFIG
from train_utils.game import selfPlay, contest
from train_utils.replay_buffer import ReplayBuffer
from utils import timeLog


class Trainer():
    def __init__(self) -> None:
        self.net = PolicyValueNet()
        self.cnt = 0  # version of best net
        self.best_net = copy.deepcopy(self.net)
        self.replay_buffer = ReplayBuffer()

    @timeLog
    def collectData(self):
        ic("collect data")
        for i in tqdm(range(TRAIN_CONFIG.game_num)):
            states, mcts_probs, values = selfPlay(self.net)
            self.replay_buffer.add(states, mcts_probs, values)

    @timeLog
    def evaluate(self):
        ic("evaluate model")
        results = {0: 0, 1: 0, -1: 0}

        for i in tqdm(range(TRAIN_CONFIG.num_contest // 2)):
            winner = contest(self.net, self.best_net)
            results[winner] += 1

        for i in tqdm(range(TRAIN_CONFIG.num_contest // 2)):
            winner = contest(self.best_net, self.net)
            winner = winner ^ 1 if winner != -1 else winner
            results[winner] += 1

        message = "result: {} win, {} lose, {} draw".format(
            results[0], results[1], results[-1])
        ic(message)
        return (results[0] + 0.5 * results[-1]) / TRAIN_CONFIG.num_contest

    def train(self):
        ic("train model")
        train_iter = self.replay_buffer.trainIter()

        for i in range(1, TRAIN_CONFIG.train_epochs + 1):
            losses, mean_loss, mean_acc = [], 0, 0
            with tqdm(total=len(train_iter)) as pbar:
                for data_batch in train_iter:
                    loss, acc = self.net.trainStep(data_batch)
                    losses.append(loss)
                    mean_loss += loss * data_batch[-1].shape[0]
                    mean_acc += acc * data_batch[-1].shape[0]
                    pbar.update()
            print("epoch {} finish".format(i))
            mean_loss /= self.replay_buffer.size()
            mean_acc /= self.replay_buffer.size()
            ic(mean_loss, mean_acc)

    def run(self):
        """[summary]
        pipeline: collect data, train, evaluate, update and repeat
        """
        for i in range(1, TRAIN_CONFIG.train_num + 1):
            # >>>>> collect data
            self.collectData()
            print("Round {} finish, buffer size {}".format(
                i, self.replay_buffer.size()))
            # save data
            if i % TRAIN_CONFIG.data_save_freq == 0:
                self.replay_buffer.save(version=str(i))

            # >>>>> train
            if self.replay_buffer.enough():
                self.train()

            # >>>>> evaluate
            if i % TRAIN_CONFIG.check_freq == 0:
                win_rate = self.evaluate()
                if win_rate >= TRAIN_CONFIG.update_threshold:
                    self.cnt += 1
                    self.best_net = copy.deepcopy(self.net)
                    self.best_net.save(version=str(self.cnt))
                    message = "new best model {}!".format(self.cnt)
                    ic(message)
                else:
                    ic("reject new model.")


if __name__ == "__main__":

    trainer = Trainer()
    trainer.run()
