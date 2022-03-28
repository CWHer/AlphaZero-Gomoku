import pickle
from typing import List, Tuple

import numpy as np
from config import ENV_CONFIG
from icecream import ic
from utils import plotHeatMap, printError


class Simulator():
    """[summary]
    Gomoku simulator

    NOTE: black == 0 / white == 1 / empty == -1
    NOTE:
        coord = (x, y)
        index = x * board_size + y
    """
    __OFFSETS = np.array(
        list(zip([0, -1, -1, -1, 0, 1, 1, 1],
                 [-1, -1, 0, 1, 1, 1, 0, -1])))

    def __init__(self) -> None:
        self.initBoard()

    def initBoard(self):
        self.board = - np.ones(
            (ENV_CONFIG.board_size, ) * 2, dtype=np.int32)
        self.turn = 0          # who is to play
        self.winner = -1       # who is winner
        self.action_log = []   # record actions taken

    @staticmethod
    def isValidPos(coord: Tuple[int, int]) -> bool:
        x, y = coord
        return (0 <= x < ENV_CONFIG.board_size
                and 0 <= y < ENV_CONFIG.board_size)

    @staticmethod
    def Idx2Coord(index: int) -> Tuple[int, int]:
        return (index // ENV_CONFIG.board_size,
                index % ENV_CONFIG.board_size)

    @staticmethod
    def Coord2Idx(coord: Tuple[int, int]) -> int:
        x, y = coord
        return x * ENV_CONFIG.board_size + y

    def plotActionProb(
            self, actions_probs: List[Tuple[int, float]]):
        """[summary]
        DEBUG function
        """
        probs = np.zeros(
            (ENV_CONFIG.board_size, ) * 2)

        for action, prob in actions_probs:
            coord = self.Idx2Coord(action)
            probs[coord] = prob

            ic.configureOutput(includeContext=True)
            printError(
                self.board[coord] != -1,
                ic.format("invalid action")
            )
            ic.configureOutput(includeContext=False)

        plotHeatMap(
            probs, "Actions-Probs", "actions_probs")

    @staticmethod
    def __chgCoord(coord, direction) -> Tuple[int, int]:
        return tuple(np.array(coord) + Simulator.__OFFSETS[direction])

    def __count(self, coord, direction, col) -> int:
        """[summary]
        count # chesses with "col" color along certain direction
        """
        num = 0
        while (self.isValidPos(coord)
               and self.board[coord] == col):
            coord = self.__chgCoord(coord, direction)
            num += 1
        return num

    # >>> load & save utils
    def load(self, file_name="board.pkl"):
        with open(file_name, "rb") as f:
            self.board = pickle.load(f)

    def save(self, file_name="board.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.board, f)

    def display(self):
        """[summary]
        beautified print
        """
        print("".join(["{:4}".format(i)
              for i in range(ENV_CONFIG.board_size)]) + "\n")
        for i in range(ENV_CONFIG.board_size):
            chesses = map(
                lambda x: ("_" if x == -1 else str(x)).center(4),
                self.board[i, :].tolist())
            print("{:<2}".format(i) + "".join(chesses) + "\n")
        print("")
    # <<< load & save utils

    # >>> MDP utils
    def getEmptyIndices(self) -> List[int]:
        """[summary]
        e.g. [0, 10, 12, 16, 20, ...]
        """
        x, y = np.where(self.board == -1)
        return [self.Coord2Idx(coord) for coord in zip(x, y)]

    def step(self, index) -> None:
        """[summary]
        step forward
        """
        coord = self.Idx2Coord(index)

        ic.configureOutput(includeContext=True)
        printError(
            self.board[coord] != -1,
            ic.format(f"duplicate step {coord} !")
        )
        ic.configureOutput(includeContext=False)

        self.board[coord] = self.turn

        # check terminal
        n_direction = len(self.__OFFSETS) // 2
        for i in range(n_direction):
            # forward and reverse directions
            cnt = self.__count(coord, i, self.turn) - 1
            cnt += self.__count(coord, i + n_direction, self.turn)

            if cnt >= ENV_CONFIG.win_cnt:
                self.winner = self.turn
                break

        self.turn ^= 1
        self.action_log.append(index)

    def backtrack(self) -> None:
        """[summary]
        backtrack a step
        """
        self.turn ^= 1
        self.winner = -1
        last_step = self.action_log.pop(-1)
        self.board[self.Idx2Coord(last_step)] = -1

    def isEnd(self) -> Tuple[bool, int]:
        """[summary]
        Returns:
            is_end (bool): [description].
            winner (int): [description]. winner (-1 for draw)
        """
        if self.winner != -1:
            return True, self.winner

        return (True, -1) if (len(self.action_log) ==
                              ENV_CONFIG.board_size ** 2) else (False, 0)
    # <<< MDP utils
