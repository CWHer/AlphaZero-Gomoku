import copy
from typing import List, Tuple

import numpy as np
from config import ENV_CONFIG, MCTS_CONFIG
from env.simulator import Simulator
from tqdm import tqdm
from utils import plotLine, timeLog

from .mcts_utils import TreeNode
from .network import PolicyValueNet
from .network_utils import ObsEncoder


class MCTS():
    """[summary]
    Monte Carlo Tree Search
    """

    def __init__(
            self, net: PolicyValueNet) -> None:
        self.net = net
        self.root = TreeNode(None, None, 1.0)

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
        while not node.isLeaf:
            node = node.select()
            env.step(node.action)

        # >>>>> expansion & simulation
        # NOTE: replace simulation with network prediction, which yields
        #    1. prior_prob for each leaf node
        #    2. v in [-1, 1] for current node
        done, winner = env.isEnd()
        # terminal
        value = 0 if winner == -1 else 1
        # non-terminal
        if not done:
            features = ObsEncoder.encode(env)
            policy, value = self.net.predict(features)
            valid_actions = env.getEmptyIndices()
            actions_probs = [
                (action, policy[action])
                for action in valid_actions
            ]
            node.expand(actions_probs)

            # if node.isRoot:
            #     env.plotActionProb(actions_probs)

        # >>>>> backpropagation
        while not node is None:
            node.update(value)
            # NOTE: opponent gets negative reward
            value = -value
            node = node.parent

    def search(self, env: Simulator) -> Tuple[List, np.ndarray]:
        # NOTE: ensure that root is correct

        # for _ in tqdm(range(MCTS_CONFIG.search_num)):
        for _ in range(MCTS_CONFIG.search_num):
            self.__search(copy.deepcopy(env))
        # self.root.display()

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
            self, net: PolicyValueNet) -> None:
        self.mcts = MCTS(net)

    def step(self, action: int) -> None:
        self.mcts.step(action)

    @timeLog
    def getAction(
        self, env: Simulator, is_train=False) -> \
            Tuple[int, Tuple[np.ndarray, np.ndarray]]:
        """[summary]
        Returns:
            (action, data), where data = (state, mcts_probs)
        """
        # NOTE: MCTS MUST NOT change env
        actions, probs = self.mcts.search(env)

        # choose action
        if is_train:
            # NOTE: add noise for exploration
            noise = np.random.dirichlet(
                MCTS_CONFIG.dirichlet_alpha * np.ones(probs.shape))
            epsilon = MCTS_CONFIG.dirichlet_eps
            noisy_probs = (1 - epsilon) * probs + epsilon * noise
            action = np.random.choice(actions, p=noisy_probs)

            # plotLine(probs.tolist(), "original probs", "probs")
            # plotLine(noisy_probs, "noisy probs", "noisy")
        else:
            action = actions[np.argmax(probs)]

        # mcts probs
        mcts_probs = np.zeros(ENV_CONFIG.board_size ** 2)
        mcts_probs[np.array(actions)] = probs

        return action, \
            (ObsEncoder.encode(env), mcts_probs)
