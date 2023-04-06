from typing import List, Tuple

import numba
import numpy as np
from icecream import ic
from utils import plotHeatMap, printError

from config import ENV_CONFIG


@numba.njit
def _isValidPos(coord: Tuple[int, int],
                board_size: int
                ) -> bool:
    x, y = coord
    return 0 <= x < board_size \
        and 0 <= y < board_size


@numba.njit
def _Idx2Coord(index: int,
               board_size: int
               ) -> Tuple[int, int]:
    return index // board_size, index % board_size


@numba.njit
def _Coord2Idx(coord: Tuple[int, int],
               board_size: int
               ) -> int:
    x, y = coord
    return x * board_size + y


@numba.njit
def _chgCoord(coord: Tuple[int, int],
              direction: int,
              ) -> Tuple[int, int]:
    dx = [0, -1, -1, -1, 0, 1, 1, 1]
    dy = [-1, -1, 0, 1, 1, 1, 0, -1]
    x, y = coord
    return x + dx[direction], y + dy[direction]


@numba.njit
def _count(coord: Tuple[int, int],
           direction: int,
           col: int,
           board_size: int,
           board: np.ndarray,
           ) -> int:
    count = 0
    while (_isValidPos(coord, board_size)
           and board[coord] == col):
        count += 1
        coord = _chgCoord(coord, direction)
    return count


@numba.njit
def _checkDone(coord: Tuple[int, int],
               col: int,
               board_size: int,
               board: np.ndarray,
               win_length: int,
               ) -> bool:
    for direction in range(4):
        count = _count(coord, direction, col, board_size, board) \
            + _count(coord, direction + 4, col, board_size, board) - 1
        if count >= win_length:
            return True
    return False


@numba.njit
def _getAllActions(board_size: int,
                   board: np.ndarray,
                   ) -> List[int]:
    x, y = np.where(board == -1)
    return [_Coord2Idx(coord, board_size) for coord in zip(x, y)]


class GobangEnv():
    def __init__(self,
                 board_size=ENV_CONFIG.board_size,
                 win_length=ENV_CONFIG.win_cnt
                 ) -> None:
        self.board_size = board_size
        self.win_length = win_length
        self.initBoard()
        self._compile()

    def _compile(self):
        _Idx2Coord(0, self.board_size)
        _Coord2Idx((0, 0), self.board_size)
        _isValidPos((0, 0), self.board_size)
        _chgCoord((0, 0), 0)
        _count((0, 0), 0, 0, self.board_size, self.board)
        _checkDone((0, 0), 0, self.board_size, self.board, self.win_length)
        _getAllActions(self.board_size, self.board)

    def initBoard(self):
        self.board = -np.ones(
            (self.board_size, ) * 2, dtype=np.int32)
        self.turn = 0          # who is to play
        self.winner = -1       # who is winner
        self.action_log = []   # record actions taken

    def plotActionProb(
            self, actions_probs: List[Tuple[int, float]]):
        """[summary]
        DEBUG function
        """
        probs = np.zeros(
            (ENV_CONFIG.board_size, ) * 2)

        for action, prob in actions_probs:
            coord = _Idx2Coord(action, self.board_size)
            probs[coord] = prob

            ic.configureOutput(includeContext=True)
            printError(
                self.board[coord] != -1,
                ic.format("invalid action")
            )
            ic.configureOutput(includeContext=False)

        plotHeatMap(probs, "Actions-Probs", 
                    "actions_probs")

    def display(self, output_size=(5, 2)):
        """[summary]
        beautified print
        """
        width, height = output_size

        def printLine(prefix, content):
            print(prefix, "".join(content), "\n" * height, end="")

        printLine(
            prefix=" " * width,
            content=[f"[{i}]".center(width)
                     for i in range(self.board_size)]
        )
        for i in range(self.board_size):
            chesses = map(
                lambda x: ("_" if x == -1 else str(x)).center(width),
                self.board[i, :].tolist())
            printLine(
                prefix=f"[{i}]".center(width),
                content=chesses)

    # >>> MDP utils
    def getAllActions(self) -> List[int]:
        """[summary]
        e.g. [0, 10, 12, 16, 20, ...]
        """
        return _getAllActions(self.board_size, self.board)

    def step(self, index) -> None:
        coord = _Idx2Coord(index, self.board_size)
        self.board[coord] = self.turn

        # check termination condition
        if _checkDone(coord, self.turn,
                      self.board_size, self.board,
                      self.win_length):
            self.winner = self.turn

        self.turn ^= 1
        self.action_log.append(index)

    def backtrack(self) -> None:
        self.turn ^= 1
        self.winner = -1
        last_step = self.action_log.pop(-1)
        self.board[_Idx2Coord(last_step, self.board_size)] = -1

    def isDone(self) -> Tuple[bool, int]:
        """[summary]
        Returns:
            is_end (bool): [description].
            winner (int): [description]. winner (-1 for draw)
        """
        if self.winner != -1:
            return True, self.winner

        return (True, -1) if (len(self.action_log) ==
                              self.board_size ** 2) else (False, 0)
    # <<< MDP utils
