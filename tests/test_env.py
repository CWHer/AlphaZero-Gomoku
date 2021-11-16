from env.simulator import Simulator
from icecream import ic

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
env.step(Simulator.Coord2Idx((1, 5)))
env.print()
ic(env.isDone())
env.backtrack(1)
