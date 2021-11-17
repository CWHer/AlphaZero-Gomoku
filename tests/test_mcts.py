from icecream import ic

from agent.network import PolicyValueNet
from agent.mcts import MCTSPlayer
from env.simulator import Simulator

env = Simulator()
env.print()
# ic(env.getEmptyIndex())
env.step(Simulator.Coord2Idx((1, 1)))
env.step(Simulator.Coord2Idx((2, 2)))
env.step(Simulator.Coord2Idx((1, 2)))
env.step(Simulator.Coord2Idx((3, 3)))
env.step(Simulator.Coord2Idx((1, 3)))
env.step(Simulator.Coord2Idx((4, 4)))
env.step(Simulator.Coord2Idx((1, 4)))
env.step(Simulator.Coord2Idx((5, 5)))
env.print()
# ic(env.isDone())
# env.backtrack(1)
player = MCTSPlayer(PolicyValueNet())
action, mcts_prob = player.getAction(env, is_train=True)
ic(env.Idx2Coord(action))
