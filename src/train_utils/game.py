import random
from multiprocessing import Queue
from typing import List

import numpy as np
import torch
from agent.batch_inference import SharedData
from agent.net_mcts import MCTSPlayer
from env.gobang_env import GobangEnv
from utils import printInfo


def selfPlay(
    seed: int, index: int,
        shared_data: SharedData,
        done_queue: Queue) -> None:
    """[summary]
    Returns:
        (states, mcts_probs, mcts_vals)
    """
    # fix seeds (torch, np, random)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = GobangEnv()
    done, winner = env.isDone()

    episode_len, data_buffer = 0, []
    players = [
        MCTSPlayer(index, shared_data)
        for _ in range(2)
    ]

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
        done, winner = env.isDone()

    shared_data.finish()
    states, mcts_probs = zip(*data_buffer)

    if winner == -1:
        printInfo("Game Over. Draw")
        done_queue.put(
            (states, mcts_probs,
             np.zeros(episode_len, dtype=np.float32)))
        return

    # printInfo(f"Game Over. Player {winner} win!")
    mcts_vals = np.array(
        [1 if (i & 1) == winner else -1
         for i in range(episode_len)], dtype=np.float32
    )
    done_queue.put(
        (states, mcts_probs, mcts_vals))


def contest(
    seed: int, index: int,
        shared_data: SharedData, done_queue: Queue,
        players: List = [None, None]) -> None:
    """[summary]
    contest between player0 and player1
    NOTE: modify players parameter to use pure MCTS

    Returns:
        winner
    """
    # fix seeds (torch, np, random)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = GobangEnv()
    done, winner = env.isDone()
    for i in range(2):
        if players[i] is None:
            players[i] = MCTSPlayer(index, shared_data)

    while not done:
        action, _ = \
            players[env.turn].getAction(env)

        env.step(action)
        # env.display()
        # update MCTS root node
        for i in range(2):
            players[i].step(action)

        # check game status
        done, winner = env.isDone()

    shared_data.finish()
    # printInfo(f"Game Over. Winner: {winner}")
    done_queue.put(winner)
