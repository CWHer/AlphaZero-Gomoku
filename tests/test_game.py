from icecream import ic

from agent.network import PolicyValueNet
from train_utils.game import selfPlay
from train_utils.replay_buffer import ReplayBuffer

net = PolicyValueNet()
states, mcts_probs, values = selfPlay(net)
buffer = ReplayBuffer()
# for state in states:
#     print(state[-1][0][0])
buffer.add(states, mcts_probs, values)
buffer.save(version="test")
train_iter = buffer.trainIter()
for data_batch in train_iter:
    loss, acc = net.trainStep(data_batch)
    ic(loss, acc)
net.save(version="test")
