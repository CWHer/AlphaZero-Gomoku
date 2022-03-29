from typing import List, Tuple

import numpy as np
import torch
from agent.mcts import MCTSPlayer
from agent.network import PolicyValueNet
from env.simulator import Simulator
from utils import printInfo


def selfPlay(net: PolicyValueNet, seed: int) -> \
        Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """[summary]
    Returns:
        (states, mcts_probs, mcts_vals)
    """
    # fix seeds (torch, np, random)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = Simulator()
    done, winner = env.isEnd()

    episode_len, data_buffer = 0, []
    players = [MCTSPlayer(net) for _ in range(2)]

    while not done:
        # NOTE: data = (states, mcts_probs)
        action, data = \
            players[env.turn].getAction(env, is_train=True)
        data_buffer.append(data)

        env.step(action)
        # env.display()
        episode_len += 1
        # update MCTS root node
        for i in range(2):
            players[i].step(action)

        # check game status
        done, winner = env.isEnd()

    states, mcts_probs = zip(*data_buffer)

    if winner == -1:
        printInfo("Game Over. Draw")
        return states, mcts_probs, \
            np.zeros(episode_len, dtype=np.float32)

    # printInfo(f"Game Over. Player {winner} win!")
    mcts_vals = np.array(
        [1 if (i & 1) == winner else -1
         for i in range(episode_len)], dtype=np.float32
    )
    return states, mcts_probs, mcts_vals


def contest(
    net0: PolicyValueNet, net1: PolicyValueNet,
        seed: int) -> int:
    """[summary]
    contest between net0 and net1

    Returns:
        winner
    """
    # fix seeds (torch, np, random)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = Simulator()
    done, winner = env.isEnd()
    players = [MCTSPlayer(net0), MCTSPlayer(net1)]

    while not done:
        action, _ = \
            players[env.turn].getAction(env)

        env.step(action)
        # env.display()
        # update MCTS root node
        for i in range(2):
            players[i].step(action)

        # check game status
        done, winner = env.isEnd()

    # printInfo(f"Game Over. Winner: {winner}")
    return winner
