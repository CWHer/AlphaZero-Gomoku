import copy
from typing import List, Tuple

import numpy as np
import tqdm
from agent.batch_inference import SharedData
from agent.net_mcts_utils import TreeNode
from agent.network_utils import ObsEncoder
from env.gobang_env import GobangEnv
from utils import plotLine, timeLog

from config import ENV_CONFIG, MCTS_CONFIG


class MCTS():
    """[summary]
    Monte Carlo Tree Search
    """

    def __init__(
        self, index: int,
            shared_data: SharedData) -> None:
        self.index = index
        self.shared_data = shared_data
        self.root = TreeNode(None, None, 1.0)

    def step(self, action: int) -> None:
        self.root = self.root.step(action)

    def _search(self, env: GobangEnv) -> None:
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
        #    2. v in [-1, 1] for current node (from current player's perspective)
        done, winner = env.isDone()
        # terminal
        # NOTE: last action is made by opponent
        value = 0 if winner == -1 else -1
        # non-terminal
        if not done:
            features = ObsEncoder.encode(env)
            self.shared_data.put(self.index, features)
            # NOTE: batch inference (on another process)
            policy, value = self.shared_data.get(self.index)

            valid_actions = env.getAllActions()
            actions_probs = [
                (action, policy[action])
                for action in valid_actions
            ]
            node.expand(actions_probs)

            # if node.isRoot:
            #     env.plotActionProb(actions_probs)
        else:
            # NOTE: forced alignment
            self.shared_data.put(self.index, None)

        # >>>>> backpropagation
        # NOTE: value is used to evaluate last action,
        #   which is made by opponent (not current player)
        while not node is None:
            node.update(-value)
            # NOTE: opponent gets negative reward
            value = -value
            node = node.parent

    def search(self, env: GobangEnv) -> Tuple[List, np.ndarray]:
        for _ in tqdm.trange(MCTS_CONFIG.n_search, disable=True):
            self._search(copy.deepcopy(env))
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
        self, index,
            shared_data: SharedData) -> None:
        self.mcts = MCTS(index, shared_data)

    def step(self, action: int) -> None:
        self.mcts.step(action)

    # @timeLog
    def getAction(
        self, env: GobangEnv, is_train=False) -> \
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
            # action = np.random.choice(actions, p=probs)

        # mcts probs
        mcts_probs = np.zeros(ENV_CONFIG.board_size ** 2)
        mcts_probs[np.array(actions)] = probs

        return action, \
            (ObsEncoder.encode(env), mcts_probs)
