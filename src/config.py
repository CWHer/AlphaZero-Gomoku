import json
import os
from collections import namedtuple
from datetime import datetime


ENV_CONFIG = {
    # board
    "win_cnt": 5,
    "board_size": 10,

    # display
    "output_size": (5, 2),
}

NETWORK_CONFIG = {
    "checkpoint_dir": "checkpoint",

    # features
    "n_periods": 4,

    # network architecture
    "n_res": 10,
    "n_channels": 128,

    # loss
    "value_weight": 1,

    # optimizer
    "learning_rate": 0.001,
    "l2_weight": 1e-4,
}

MCTS_CONFIG = {
    "c_uct": 2,
    "c_puct": 5,
    "n_search": 800,

    # action selection
    "inv_temp": 1 / 1,
    "dirichlet_alpha": 0.3,
    "dirichlet_eps": 0.25,
}

DATA_CONFIG = {
    "dataset_dir": "dataset",
    "save_freq": 200,
    "augment_data": True,
    "replay_size": 500000,

    # sample
    "train_threshold": 12000,
    "sample_size": 512 * 20,
    "batch_size": 512,
}

TRAIN_CONFIG = {
    "parameters_dir": "parameters",
    "log_dir": "logs/",

    "epochs": 5000,
    # multiprocessing
    "n_process": 20,

    # self play
    "n_game": 20,

    # evaluate
    "eval_freq": 20,
    "update_thr": 0.55,     # against best net
    "n_contest": 20,
    # "update_thr": 0.75,      # against pure mcts
    # "dn_search": 1000,
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
