import copy

import numpy as np
from config import MCTS_CONFIG, MDP_CONFIG
from icecream import ic
from tqdm import tqdm
from utils import plotHeatMap, plotLines, timeLog

from .mcts_utils import TreeNode
from .network_utils import ObsEncoder


class MCTS():
    """[summary]
    Monte Carlo Tree Search
    """

    def __init__(self, net) -> None:
        self.net = net
        self.root = TreeNode(None, None, 1.0)

    @staticmethod
    def plotActionProb(env, actions_probs):
        probs = np.zeros((MDP_CONFIG.board_size, ) * 2)
        for action, prob in actions_probs:
            probs[env.Idx2Coord(action)] = prob
        plotHeatMap(probs, "none")

    def update(self, action):
        """[summary]
        update root

        """
        self.root = self.root.transfer(action)
        if not self.root is None:
            self.root.parent = None
        else:
            self.root = TreeNode(None, None, 1.0)

    def __search(self, env):
        """[summary]
        perform one search, which contains
        1. selection     2. expansion
        3. simulation    4. backpropagate
        """
        # >>>>> selection
        node = self.root
        while not node.isLeaf:
            node = node.select()
            action = node.action
            env.step(action)

        # >>>>> expansion & simulation
        # NOTE: use network to predict new node
        #   which yields prior_prob for each leaf node
        #    and v in [-1, 1] for current node
        done, winner = env.isDone()
        if not done:
            policy, value = self.net.predict(ObsEncoder.encode(env))
            legal_actions = env.getEmptyIndex()
            actions_probs = [
                (action, policy[action])
                for action in legal_actions]

            # debug
            # if node.isRoot:
            #     self.plotActionProb(env, actions_probs)

            node.expand(actions_probs)
        else:
            # NOTE: turn is one step ahead
            # ic(env.turn)
            # env.print()
            value = 0 if winner == -1 else 1

        # >>>>> backpropagate
        while node != None:
            node.update(value)
            # NOTE: negate value each layer
            value = -value
            node = node.parent

        # debug
        # self.root.printDebugInfo()

    def search(self, env):
        # NOTE: make sure root node is correct
        # for _ in tqdm(range(MCTS_CONFIG.search_num)):
        for _ in range(MCTS_CONFIG.search_num):
            self.__search(copy.deepcopy(env))

        # debug
        # self.root.printDebugInfo()

        actions, vis_cnt = list(zip(
            *[(child.action, child.getVisCount())
              for child in self.root.children]))
        # NOTE: TEMPERATURE controls the level of exploration
        #   N ^ (1 / T) = exp(1 / T * log(N))
        probs = MCTS_CONFIG.inv_temperature * \
            np.log(np.array(vis_cnt) + 1e-10)
        probs = np.exp(probs - np.max(probs))
        probs /= np.sum(probs)

        # debug
        # self.plotActionProb(env, zip(actions, probs))

        return actions, probs


class MCTSPlayer():
    def __init__(self, net) -> None:
        self.mcts = MCTS(net)

    def updateRoot(self, action):
        self.mcts.update(action)

    # @timeLog
    def getAction(self, env, is_train=False):
        """[summary]
        use MCTS to get action

        Returns:
            (action, mcts_prob)
        """
        # NOTE: search MUST NOT change env
        actions, probs = self.mcts.search(env)

        if is_train:
            # NOTE: add noise for exploration
            noise = np.random.dirichlet(
                np.full(len(probs), MCTS_CONFIG.dirichlet_alpha))
            action = np.random.choice(
                actions,
                p=((1 - MCTS_CONFIG.dirichlet_eps) * probs
                   + MCTS_CONFIG.dirichlet_eps * noise))

            # debug
            # plotLines([(probs, "original"),
            #            (((1 - MCTS_CONFIG.dirichlet_eps) * probs
            #              + MCTS_CONFIG.dirichlet_eps * noise), "adding noise")])
        else:
            # NOTE: almost equivalent to argmax
            # action = np.random.choice(actions, p=probs)
            action = actions[np.argmax(probs)]

        mcts_prob = np.zeros(MDP_CONFIG.board_size ** 2)
        mcts_prob[np.array(actions)] = probs

        # debug
        # plotHeatMap(mcts_prob.reshape(
        #     (MDP_CONFIG.board_size, ) * 2),
        #     "none")

        return action, (ObsEncoder.encode(env), mcts_prob)
