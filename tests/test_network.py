import unittest

from agent.network import PolicyValueNet
from agent.network_utils import ObsEncoder
from env.gobang_env import GobangEnv, _Coord2Idx
from icecream import ic
from torchsummary import summary
from utils import plotSparseMatrix


class TestNetwork(unittest.TestCase):
    def testNetwork(self):
        net = PolicyValueNet()
        net.setDevice()

        # summary(net.net, (11, 10, 10), batch_size=512)

        env = GobangEnv()
        # ic(env.getEmptyIndices())
        env.step(_Coord2Idx((1, 1), env.board_size))
        env.step(_Coord2Idx((2, 2), env.board_size))
        env.step(_Coord2Idx((1, 2), env.board_size))
        env.step(_Coord2Idx((3, 3), env.board_size))
        env.step(_Coord2Idx((1, 3), env.board_size))
        env.display()
        features = ObsEncoder.encode(env)
        # for i in range(features.shape[0]):
        #     plotSparseMatrix(features[i])
        policy, value = net.predict(features)
        ic(policy, value)
