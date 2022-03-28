from __future__ import annotations

import math
from typing import List, Optional, Tuple

from config import MCTS_CONFIG
from icecream import ic
from utils import printError, printInfo


class PUCT():
    """[summary]
    UCT with policy
    """

    def __init__(
            self, prior_prob: float) -> None:
        self.Q, self.N = 0, 0
        self.prob = prior_prob
        self.c_puct = MCTS_CONFIG.c_puct

    def update(self, v: float) -> None:
        self.N += 1
        self.Q += (v - self.Q) / self.N

    def U(self, Np) -> float:
        return self.prob * math.sqrt(Np) / (self.N + 1)

    def PUCT(self, Np) -> float:
        return self.Q + self.c_puct * self.U(Np)


class TreeNode():
    """[summary]
    node of MCTS

    NOTE: 
        1. node is either unvisited or fully expanded
        2. action = x * board_size + y, i.e. == index 
    """

    def __init__(
            self, parent: Optional[TreeNode],
            action: Optional[int], prior_prob: float) -> None:
        """[summary]
        s (parent) --> a (action) --> s' (self)
        """
        self.parent = parent
        self.action = action
        self.puct = PUCT(prior_prob)
        self.children: Optional[List[TreeNode]] = None

    @property
    def isLeaf(self) -> bool:
        return self.children is None

    @property
    def isRoot(self) -> bool:
        return self.parent is None

    def getVisCount(self) -> float:
        return self.puct.N

    def PUCT(self, Np) -> float:
        return self.puct.PUCT(Np)

    def update(self, v: float) -> None:
        self.puct.update(v)

    def select(self) -> TreeNode:
        """[summary]
        select child with maximum PUCT
        """
        ic.configureOutput(includeContext=True)
        printError(self.isLeaf, "nothing to select")
        ic.configureOutput(includeContext=False)

        Np = self.getVisCount()
        return max(
            self.children, key=lambda x: x.PUCT(Np))

    def expand(self, actions_probs: List[Tuple[int, float]]):
        """[summary]
        fully expand self with prior probs
        """
        self.children = [
            TreeNode(self, action, prior_prob)
            for action, prior_prob in actions_probs
        ]

    def step(self, action: int) -> TreeNode:
        """[summary]
        transit to next state
        """
        if self.isLeaf:
            return TreeNode(None, None, 1.0)

        next_state = None
        for child in self.children:
            if child.action == action:
                next_state = child
                break
        if not next_state is None:
            next_state.parent = None
            return next_state

        ic.configureOutput(includeContext=True)
        printError(True, ic.format("fail to find child!"))
        ic.configureOutput(includeContext=False)

    def display(self):
        """[summary]
        DEBUG function
        """
        Np = self.getVisCount()
        msg = "Action: {:>3} with N: {:>4}, Q: {:>+.4f}, PUCT: {:>+.4f}"
        for child in self.children:
            printInfo(msg.format(
                child.action, child.getVisCount(),
                child.puct.Q, child.PUCT(Np)))
