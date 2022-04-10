import copy
import random
from typing import List, Tuple

import numpy as np
from config import MCTS_CONFIG
from env.simulator import Simulator
from tqdm import tqdm
from utils import timeLog

from .mcts_utils import TreeNode


class MCTS():
    """[summary]
    Monte Carlo Tree Search
    """

    def __init__(self, n_search) -> None:
        self.n_search = n_search
        self.root = TreeNode(None, None)

    def step(self, action: int) -> None:
        self.root = self.root.step(action)

    def __search(self, env: Simulator) -> None:
        """[summary]
        NOTE: __search() contains:
            1. selection     2. expansion
            3. simulation    4. backpropagation
        """
        # >>>>> selection
        node = self.root
        while node.isFullyExpanded:
            node = node.select()
            env.step(node.action)

        # >>>>> expansion
        done, _ = env.isEnd()
        if not done:
            # update valid actions
            if node.valid_actions is None:
                node.updUnvisited(env.getEmptyIndices())
            node = node.expand()
            env.step(node.action)

        # >>>>> simulation
        def runRollout(env: Simulator) -> int:
            """[summary]    
            NOTE: this would change env
            """
            done, winner = env.isEnd()
            while not done:
                valid_actions = env.getEmptyIndices()
                env.step(random.choice(valid_actions))
                done, winner = env.isEnd()
            return winner

        last_turn = env.turn ^ 1
        winner = runRollout(env)
        value = 0 if winner == -1 else int(winner == last_turn)

        # >>>>> backpropagation
        while not node is None:
            node.update(value)
            # NOTE: opponent gets negative reward
            value = -value
            node = node.parent

    def search(self, env: Simulator) -> Tuple[List, np.ndarray]:
        # NOTE: ensure that root is correct

        # for _ in tqdm(range(self.n_search)):
        for _ in range(self.n_search):
            self.__search(copy.deepcopy(env))
        # self.root.display(env)

        actions, vis_cnt = list(
            zip(*[(child.action, child.getVisCount())
                  for child in self.root.children])
        )

        # NOTE: TEMPERATURE controls exploration level
        #   N ^ (1 / T) = exp(1 / T * log(N))
        logits = MCTS_CONFIG.inv_temp * \
            np.log(np.array(vis_cnt) + 1e-10)
        logits = np.exp(logits - np.max(logits))
        probs = logits / np.sum(logits)

        # env.plotActionProb(zip(actions, probs))
        return actions, probs


class MCTSPlayer():
    def __init__(
            self, n_search=MCTS_CONFIG.n_search) -> None:
        self.mcts = MCTS(n_search)

    def step(self, action: int) -> None:
        self.mcts.step(action)

    # @timeLog
    def getAction(self, env: Simulator) -> int:
        """[summary]
        Returns: action
        """
        # NOTE: MCTS MUST NOT change env
        actions, probs = self.mcts.search(env)

        # choose action
        # action = np.random.choice(actions, p=probs)
        action = actions[np.argmax(probs)]
        return action
