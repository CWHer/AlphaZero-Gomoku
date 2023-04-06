from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent.network_utils import ResidualBlock, conv3x3
from torch.cuda.amp import GradScaler, autocast
from utils import printError, printInfo

from config import ENV_CONFIG, NETWORK_CONFIG


class AZNetwork(nn.Module):
    """[summary]
    toy AlphaZero network
    """

    def __init__(self):
        super().__init__()

        in_channels = NETWORK_CONFIG.n_periods * 2 + 1
        n_channels = NETWORK_CONFIG.n_channels
        n_actions = ENV_CONFIG.board_size ** 2

        # common layers
        res_list = [ResidualBlock(in_channels, n_channels)] + \
            [ResidualBlock(n_channels, n_channels)
             for _ in range(NETWORK_CONFIG.n_res - 1)]
        self.common_net = nn.Sequential(*res_list)

        # policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(n_channels, 4, kernel_size=1),
            nn.Flatten(), nn.ReLU(),
            nn.Linear(4 * n_actions, n_actions)
        )

        # value head
        self.value_head = nn.Sequential(
            nn.Conv2d(n_channels, 2, kernel_size=1),
            nn.Flatten(), nn.ReLU(),
            nn.Linear(2 * n_actions, 256),
            nn.ReLU(), nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.common_net(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class PolicyValueNet():
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.net = AZNetwork().to(self.device)

        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=NETWORK_CONFIG.learning_rate,
            weight_decay=NETWORK_CONFIG.l2_weight
        )
        self.scaler = GradScaler()

    def setDevice(self, loc: str = "cuda:0"):
        self.device = torch.device(loc)
        self.net.to(self.device)

    def save(self, version="default"):
        checkpoint_dir = NETWORK_CONFIG.checkpoint_dir

        import os
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        printInfo(f"save network & optimizer / version({version})")
        torch.save(
            self.net.state_dict(),
            checkpoint_dir + f"/model_{version}")
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir + f"/optimizer_{version}")

    def load(self, model_path, optimizer_path=None):
        printInfo(f"load network {model_path}")
        self.net.load_state_dict(
            torch.load(model_path, map_location=self.device))

        if not optimizer_path is None:
            printInfo(f"load optimizer {optimizer_path}")
            self.optimizer.load_state_dict(torch.load(optimizer_path))

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.net.eval()

        batch_inference = True
        if features.ndim < 4:
            batch_inference = False
            features = np.expand_dims(features, 0)

        features = torch.from_numpy(features).to(self.device)
        # with autocast():
        with torch.no_grad():
            logits, value = self.net(features)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            value = value.squeeze(dim=-1).cpu().numpy()

        if not batch_inference:
            probs, value = map(
                lambda x: x.squeeze(axis=0), [probs, value])
        return probs, value

    def trainStep(self, data_batch: List[torch.Tensor]) -> Tuple[float, float]:
        """[summary]
        Returns:
            loss (float): [description]
            accuracy (float): [description]
        """
        self.net.train()
        states, mcts_probs, mcts_vals = \
            map(lambda x: x.to(self.device), data_batch)

        with autocast():
            logits, values = self.net(states)

            # accuracy
            with torch.no_grad():
                actions = logits.argmax(dim=-1)
                expert_actions = mcts_probs.argmax(dim=-1)
                accuracy = (actions == expert_actions).float().mean().item()

            # loss function: (z - v) ^ 2 - pi ^ T * log(p) + c || theta || ^ 2
            value_loss = F.mse_loss(values.view(-1), mcts_vals)
            policy_loss = F.cross_entropy(logits, mcts_probs)
            loss = NETWORK_CONFIG.value_weight * value_loss + policy_loss
            printError(torch.isnan(loss), "loss is nan")

        # DEBUG: torchviz
        # from torchviz import make_dot
        # dot = make_dot(loss)
        # dot.format = "png"
        # dot.render("loss")

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), accuracy
