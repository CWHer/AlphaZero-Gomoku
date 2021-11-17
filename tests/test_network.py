import torch
from icecream import ic
from torchsummary import summary

from agent.network import PolicyValueNet
from agent.network_utils import ObsEncoder
from env.simulator import Simulator
from utils import plotSparseMatrix

net = PolicyValueNet()
# net.setDevice(torch.device("cpu"))
# summary(net.net, (11, 15, 15), batch_size=512)

env = Simulator()
# ic(env.getEmptyIndex())
env.step(Simulator.Coord2Idx((1, 1)))
env.step(Simulator.Coord2Idx((2, 2)))
env.step(Simulator.Coord2Idx((1, 2)))
env.step(Simulator.Coord2Idx((3, 3)))
env.step(Simulator.Coord2Idx((1, 3)))
env.print()
features = ObsEncoder.encode(env)
# for i in range(features.shape[0]):
#     plotSparseMatrix(features[i], "none")
policy, value = net.predict(features)
ic(policy, value)
