from icecream import ic

from env.simulator import Simulator


def testNetMCTS(env):
    from multiprocessing import Process

    from agent.batch_inference import SharedData, batchInference
    from agent.net_mcts import MCTSPlayer
    from agent.network import PolicyValueNet

    def func(env, index, shared_data):
        player = MCTSPlayer(index, shared_data)
        action, mcts_prob = player.getAction(env, is_train=True)
        ic(env.Idx2Coord(action))
        shared_data.finish()

    net = PolicyValueNet()
    net.setDevice()

    with SharedData(n_proc=1) as shared_data:
        shared_data.reset()
        proc = Process(target=func, args=(env, 0, shared_data))
        proc.start()

        batchInference(shared_data, net)
        proc.join()


def testMCTS(env):
    from agent.mcts import MCTSPlayer

    player = MCTSPlayer(n_search=4000)
    action, _ = player.getAction(env)
    ic(env.Idx2Coord(action))


env = Simulator()
env.step(Simulator.Coord2Idx((1, 1)))
env.step(Simulator.Coord2Idx((2, 2)))
env.step(Simulator.Coord2Idx((1, 2)))
env.step(Simulator.Coord2Idx((3, 3)))
env.step(Simulator.Coord2Idx((1, 3)))
env.step(Simulator.Coord2Idx((4, 4)))
env.step(Simulator.Coord2Idx((1, 4)))
env.step(Simulator.Coord2Idx((5, 5)))
ic(len(env.getEmptyIndices()))
env.display()
ic(env.isEnd())
# env.backtrack()

testMCTS(env)
# testNetMCTS(env)

print("Done")
