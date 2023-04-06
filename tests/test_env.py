import unittest

from env.gobang_env import GobangEnv, _Coord2Idx


class TestGobangEnv(unittest.TestCase):
    def testGobangEnv(self):
        env = GobangEnv()
        # ic(env.getEmptyIndex())
        env.step(_Coord2Idx((1, 1), env.board_size))
        env.step(_Coord2Idx((2, 2), env.board_size))
        env.step(_Coord2Idx((1, 2), env.board_size))
        env.step(_Coord2Idx((3, 3), env.board_size))
        env.step(_Coord2Idx((1, 3), env.board_size))
        env.step(_Coord2Idx((4, 4), env.board_size))
        env.step(_Coord2Idx((1, 4), env.board_size))
        env.step(_Coord2Idx((5, 5), env.board_size))
        env.step(_Coord2Idx((1, 5), env.board_size))
        env.display()
        done, winner = env.isDone()
        self.assertTrue(done)
        env.backtrack()
        done, winner = env.isDone()
        self.assertFalse(done)
        env.display()
