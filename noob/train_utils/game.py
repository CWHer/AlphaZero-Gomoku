from env.simulator import Simulator
from agent.mcts import MCTSPlayer
from icecream import ic


def selfPlay(net):
    """[summary]
    self play and gather experiences
    Args:
        net ([type]): [description]
    Returns:
        states [List[np.ndarray]]: [description]
        mcts_probs [List[np.ndarray]]: [description]
        values [List[float]]: [description]
    """
    # TODO: initialize seeds (torch, np, random)
    env = Simulator()

    data_buffer, episode_len = [], 0
    players = [MCTSPlayer(net) for _ in range(2)]

    while True:
        episode_len += 1

        # NOTE: getAction MUST NOT change env
        # NOTE: data = (features(states), mcts_probs)
        action, data = players[env.turn].getAction(env, is_train=True)
        data_buffer.append(data)

        # take action
        env.step(action)
        # update root
        for i in range(2):
            players[i].updateRoot(action)

        # debug
        # env.print()

        # check game status
        done, winner = env.isDone()

        if done:
            # FIXME: NOTE: drop data of draws
            if winner == -1:
                ic("Game Over. Draw")
                return [], [], []
            message = "Game Over. Player{} win!".format(winner)
            # ic(message)

            states, mcts_probs = zip(*data_buffer)
            values = [
                1 if (i & 1) == winner else -1
                for i in range(episode_len)]
            # ic(len(states), len(values))

            return states, mcts_probs, values


def contest(net0, net1):
    """[summary]
    contest between net0 and net1

    Returns:
        int: [description]. winner
    """
    # TODO: initialize seeds (torch, np, random)
    env = Simulator()
    players = [MCTSPlayer(net0), MCTSPlayer(net1)]

    while True:
        # NOTE: getAction MUST NOT change env
        action, _ = players[env.turn].getAction(env)

        # take action
        env.step(action)
        # update root
        for i in range(2):
            players[i].updateRoot(action)

        # debug
        # env.print()

        # check game status
        done, winner = env.isDone()
        if done:
            message = "Game Over. {}".format(
                "Draw" if winner == -1
                else "Player{} win!".format(winner))
            # ic(message)
            return winner
