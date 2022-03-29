from icecream import ic

from agent.network import PolicyValueNet
from train_utils.game import contest, selfPlay
from train_utils.replay_buffer import ReplayBuffer


net = PolicyValueNet()
states, mcts_probs, mcts_vals = selfPlay(net, seed=0)

buffer = ReplayBuffer()
buffer.add(states, mcts_probs, mcts_vals)
buffer.save()
ic(len(buffer))
for i in range(50):
    train_iter = buffer.sample()
    for data_batch in train_iter:
        loss, acc = net.trainStep(data_batch)
        ic(loss, acc)
net.save()

# best_net = PolicyValueNet()
# best_net.load("xxxx")
# winner = contest(best_net, net, seed=0)
