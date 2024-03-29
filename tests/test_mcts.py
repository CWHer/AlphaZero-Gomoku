import unittest

from env.gobang_env import GobangEnv, _Coord2Idx
from icecream import ic


class TestGobangMCTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = GobangEnv()
        env.step(_Coord2Idx((1, 1), env.board_size))
        env.step(_Coord2Idx((2, 2), env.board_size))
        env.step(_Coord2Idx((1, 2), env.board_size))
        env.step(_Coord2Idx((3, 3), env.board_size))
        env.step(_Coord2Idx((1, 3), env.board_size))
        env.step(_Coord2Idx((4, 4), env.board_size))
        env.step(_Coord2Idx((1, 4), env.board_size))
        env.display()

        cls.env = env

    def testNetMCTS(self):
        from multiprocessing import Process

        from agent.batch_inference import SharedData, batchInference
        from agent.net_mcts import MCTSPlayer
        from agent.network import PolicyValueNet

        def func(env, index, shared_data):
            player = MCTSPlayer(index, shared_data)
            action, mcts_prob = player.getAction(env, is_train=True)
            ic(action)
            shared_data.finish()

        net = PolicyValueNet()
        net.setDevice()

        with SharedData(n_proc=1) as shared_data:
            shared_data.reset()
            proc = Process(target=func, args=(self.env, 0, shared_data))
            proc.start()

            batchInference(shared_data, net)
            proc.join()

    def testMCTS(self):
        from agent.mcts import MCTSPlayer

        player = MCTSPlayer(n_search=4000)
        action, _ = player.getAction(self.env)
        ic(action)
