from itertools import product

import numpy as np
from config import MDP_CONFIG
from icecream import ic
from utils import printError


class Simulator():
    """[summary]
    Gomoku simulator

    NOTE: black: 0 / white: 1 / empty: -1
    NOTE: coord = (x, y)  index = x * board_size + y
    """
    board_size = MDP_CONFIG.board_size
    DELTA = np.array(
        list(zip([0, -1, -1, -1, 0, 1, 1, 1],
                 [-1, -1, 0, 1, 1, 1, 0, -1])))

    def __init__(self) -> None:
        self.initBoard()

    def initBoard(self):
        self.board = -np.ones(
            (self.board_size, ) * 2, dtype=int)
        self.winner = -1    # who is winner
        self.turn = 0       # who is to play
        self.actions = []   # record actions taken

    @staticmethod
    def isValidPos(coord) -> bool:
        x, y = coord
        return (x >= 0 and x < Simulator.board_size
                and y >= 0 and y < Simulator.board_size)

    @staticmethod
    def Idx2Coord(index) -> tuple:
        return (index // Simulator.board_size,
                index % Simulator.board_size)

    @staticmethod
    def Coord2Idx(coord) -> int:
        x, y = coord
        return x * Simulator.board_size + y

    @staticmethod
    def chgCoord(coord, direction):
        return tuple(np.array(coord) + Simulator.DELTA[direction])

    def __count(self, coord, direction, col) -> int:
        """[summary]
        count # chesses with same color along certain direction
        """
        ret = 0
        while (self.isValidPos(coord)
               and self.board[coord] == col):
            coord = self.chgCoord(coord, direction)
            ret += 1
        return ret

    def read(self, file_name="board.txt"):
        """[summary]
        format: _ 0 _ 0 _ 1 _ _ _ ...
        """
        with open(file_name, "r") as f:
            for i in range(self.board_size):
                line = f.readline().replace("_", "-1").split(" ")
                self.board[i, :] = np.array(list(map(int, line)))
        # self.print()

    def print(self):
        """[summary]
        format: _ 0 _ 0 _ 1 _ _ _ ...
        """
        for coord in product(
                range(self.board_size), repeat=2):
            col = self.board[coord]
            print("{} ".format(
                "_" if col == -1 else col),
                end=" " if coord[-1] != self.board_size - 1 else "\n")
        print("")

    def getEmptyIndex(self):
        """[summary]
        return a list of empty indices (NOT coordinates)
        e.g. [0, 10, 12, 16, 20, ...]
        """
        x, y = np.where(self.board == -1)
        return [self.Coord2Idx(coord) for coord in zip(x, y)]

    def step(self, index):
        """[summary]
        put a chess

        NOTE: this would change Simulator internal variables
        """
        coord = self.Idx2Coord(index)
        ic.configureOutput(includeContext=True)
        printError(
            self.board[coord] != -1,
            f"try to repeat a existent step {coord} !")
        ic.configureOutput(includeContext=False)
        self.board[coord] = self.turn

        # check terminal
        for i in range(4):
            c1 = self.__count(coord, i, self.turn)
            # opposite direction
            c2 = self.__count(coord, i + 4, self.turn)
            if c1 + c2 - 1 >= MDP_CONFIG.win_length:
                self.winner = self.turn
                break

        self.turn = self.turn ^ 1
        self.actions.append(index)

    def backtrack(self, index):
        """[summary]
        backtrack a step

        NOTE: this would change Simulator internal variables
        """
        ic.configureOutput(includeContext=True)
        printError(
            self.actions[-1] != index,
            f"try to backtrack a non-existent step {index} !"
        )
        ic.configureOutput(includeContext=False)

        self.turn = self.turn ^ 1
        self.actions.pop(-1)

        self.board[self.Idx2Coord(index)] = -1
        self.winner = -1

    def isDone(self):
        """[summary]

        Returns:
            game_status (bool): [description]. whether game is done
            winner (int): [description]. winner (-1 for draw)
        """
        if self.winner != -1:
            return True, self.winner
        if len(self.actions) == self.board_size ** 2:
            return True, -1  # draw
        return False, 0
