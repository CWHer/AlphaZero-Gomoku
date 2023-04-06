from __future__ import annotations

import math
import random
from typing import List, Optional

from icecream import ic
from utils import printError, printInfo

from config import MCTS_CONFIG


class UCT():
    def __init__(self) -> None:
        self.Q, self.N = 0, 0
        self.c_uct = MCTS_CONFIG.c_uct

    def update(self, v: float) -> None:
        self.N += 1
        self.Q += (v - self.Q) / self.N

    def U(self, Np) -> float:
        return math.sqrt(math.log(Np) / self.N)

    def UCT(self, Np) -> float:
        return self.Q + self.c_uct * self.U(Np)


class TreeNode():
    """[summary]
    node of MCTS

    NOTE: action = x * board_size + y, i.e. == index 
    """

    def __init__(
            self, parent: Optional[TreeNode],
            action: Optional[int]) -> None:
        """[summary]
        s (parent) --> a (action) --> s' (self)
        """
        self.parent = parent
        self.action = action
        self.uct = UCT()
        self.children: List[TreeNode] = []
        self.valid_actions: Optional[List[int]] = None

    def updUnvisited(self, valid_actions) -> None:
        self.valid_actions = valid_actions
        random.shuffle(self.valid_actions)

    @property
    def isFullyExpanded(self) -> bool:
        """[summary]
        NOTE: treat terminal node as not fully expanded one
        """
        return not self.valid_actions \
            and not self.valid_actions is None

    @property
    def isRoot(self) -> bool:
        return self.parent is None

    def getVisCount(self) -> float:
        return self.uct.N

    def UCT(self, Np) -> float:
        return self.uct.UCT(Np)

    def update(self, v: float) -> None:
        self.uct.update(v)

    def select(self) -> TreeNode:
        """[summary]
        select child with maximum UCT
        """
        ic.configureOutput(includeContext=True)
        printError(not self.children, "nothing to select")
        ic.configureOutput(includeContext=False)

        Np = self.getVisCount()
        return max(
            self.children, key=lambda x: x.UCT(Np))

    def expand(self) -> TreeNode:
        """[summary]
        uniformly randomly choose one possible action and expand
        """
        ic.configureOutput(includeContext=True)
        printError(self.valid_actions is None, "try updUnvisited()!")
        printError(self.isFullyExpanded, "nothing to expand")
        ic.configureOutput(includeContext=False)

        action = self.valid_actions.pop()
        child = TreeNode(self, action)
        self.children.append(child)
        return child

    def step(self, action: int) -> TreeNode:
        """[summary]
        transit to next state
        """
        next_state = None
        for child in self.children:
            if child.action == action:
                next_state = child
                break
        if not next_state is None:
            next_state.parent = None
            return next_state

        return TreeNode(None, None)

    def display(self):
        """[summary]
        DEBUG function
        """
        Np = self.getVisCount()
        msg = "Action: {} with N: {:>4}, Q: {:>+.4f}, UCT: {:>+.4f}"
        for child in self.children:
            printInfo(
                msg.format(child.action, child.getVisCount(),
                           child.uct.Q, child.UCT(Np))
            )
