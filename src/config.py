import json
import os
from collections import namedtuple
from datetime import datetime


ENV_CONFIG = {
    # board
    "win_cnt": 5,
    "board_size": 15,

    # display
    "output_size": (5, 2),
}

NETWORK_CONFIG = {
    "checkpoint_dir": "checkpoint",

    # features
    "n_periods": 2,

    # network architecture
    "n_res": 4,
    "n_channels": 256,

    # loss
    "value_weight": 1,

    # optimizer
    "learning_rate": 0.001,
    "l2_weight": 1e-4,
}

MCTS_CONFIG = {
    "c_puct": 5,
    "n_search": 1600,

    # action selection
    "inv_temp": 1 / 1,
    "dirichlet_alpha": 0.3,
    "dirichlet_eps": 0.1,
}

DATA_CONFIG = {
    "dataset_dir": "dataset",
    "save_freq": 100,
    "augment_data": True,
    "replay_size": 20000,

    # sample
    "train_threshold": 2000,
    "sample_size": 512 * 2,
    "batch_size": 512,
}

TRAIN_CONFIG = {
    "parameters_dir": "parameters",
    "log_dir": "logs/",

    "epochs": 10000,
    # multiprocessing
    "n_process": 2,

    # self play
    "n_game": 2,

    # evaluate
    "eval_freq": 10,
    "update_thr": 0.55,
    "n_contest": 20,
}


def saveSettings():
    para_dir = TRAIN_CONFIG["parameters_dir"]
    if not os.path.exists(para_dir):
        os.mkdir(para_dir)

    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    TRAIN_CONFIG["log_dir"] += timestamp
    file_path = para_dir + f"/para_{timestamp}.json"
    with open(file_path, "w") as f:
        json.dump(
            {"ENV_CONFIG": ENV_CONFIG,
             "NETWORK_CONFIG": NETWORK_CONFIG,
             "MCTS_CONFIG": MCTS_CONFIG,
             "DATA_CONFIG": DATA_CONFIG,
             "TRAIN_CONFIG": TRAIN_CONFIG},
            f, indent=4)


saveSettings()


ENV_CONFIG_TYPE = namedtuple("ENV_CONFIG", ENV_CONFIG.keys())
ENV_CONFIG = ENV_CONFIG_TYPE._make(ENV_CONFIG.values())

NETWORK_CONFIG_TYPE = namedtuple("NETWORK_CONFIG", NETWORK_CONFIG.keys())
NETWORK_CONFIG = NETWORK_CONFIG_TYPE._make(NETWORK_CONFIG.values())

MCTS_CONFIG_TYPE = namedtuple("MCTS_CONFIG", MCTS_CONFIG.keys())
MCTS_CONFIG = MCTS_CONFIG_TYPE._make(MCTS_CONFIG.values())

DATA_CONFIG_TYPE = namedtuple("DATA_CONFIG", DATA_CONFIG.keys())
DATA_CONFIG = DATA_CONFIG_TYPE._make(DATA_CONFIG.values())

TRAIN_CONFIG_TYPE = namedtuple("TRAIN_CONFIG", TRAIN_CONFIG.keys())
TRAIN_CONFIG = TRAIN_CONFIG_TYPE._make(TRAIN_CONFIG.values())
