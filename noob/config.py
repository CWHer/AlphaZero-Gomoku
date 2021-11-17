from collections import namedtuple

# NOTE: dictionaries are transformed to namedtuples

MDP_CONFIG = {
    "board_size": 10,
    "win_length": 5,
}


NETWORK_CONFIG = {
    "periods_num": 5,
    "num_channels": 256,
    "num_res": 4,
}

MCTS_CONFIG = {
    "inv_temperature": 1/1,
    "search_num": 1600,
    "c_puct": 5,
    "dirichlet_alpha": 0.3,
    "dirichlet_eps": 0.1,
}

TRAIN_CONFIG = {
    "train_epochs": 5,
    "c_loss": 1,
    "l2_weight": 1e-4,
    "learning_rate": 0.001,
    "checkpoint_dir": "checkpoint",
    "batch_size": 512,
    "train_threshold": 10000,
    "replay_size": 1000000,
    "dataset_dir": "dataset",
    "data_save_freq": 50,

    "train_num": 10000,
    "check_freq": 20,
    "update_threshold": 0.55,
    "num_contest": 20,

    "game_num": 30,
    "processes_num": 30,
}

MDP_CONFIG_TYPE = namedtuple("MDP_CONFIG", MDP_CONFIG.keys())
MDP_CONFIG = MDP_CONFIG_TYPE._make(MDP_CONFIG.values())

NETWORK_CONFIG_TYPE = namedtuple("NETWORK_CONFIG", NETWORK_CONFIG.keys())
NETWORK_CONFIG = NETWORK_CONFIG_TYPE._make(NETWORK_CONFIG.values())

MCTS_CONFIG_TYPE = namedtuple("MCTS_CONFIG", MCTS_CONFIG.keys())
MCTS_CONFIG = MCTS_CONFIG_TYPE._make(MCTS_CONFIG.values())

TRAIN_CONFIG_TYPE = namedtuple("TRAIN_CONFIG", TRAIN_CONFIG.keys())
TRAIN_CONFIG = TRAIN_CONFIG_TYPE._make(TRAIN_CONFIG.values())
