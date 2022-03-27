import sys
import time
from functools import wraps
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def timeLog(func):
    @wraps(func)
    def clocked(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        print("{} finishes after {:.2f} s".format(
            func.__name__, time.time() - start_time))
        return ret
    return clocked


# >>> log utils
def printInfo(msg):
    print("[INFO]: {}".format(msg))


def printWarn(expr: bool, msg):
    if expr:
        print("[WARN]: {}".format(msg))


def printError(expr: bool, msg):
    if expr:
        print("[ERROR]: {}".format(msg))
        sys.exit(1)
# <<< log utils


# >>> plot utils
def plotLine(
        data: List, title: str = "default",
        name: str = "default"):
    fig, ax = plt.subplots()
    ax.plot(range(0, len(data)), data)
    ax.set_title(title)
    fig.savefig(name)


def plotSemilogy(
        data: List, title: str = "default",
        name: str = "default"):
    fig, ax = plt.subplots()
    ax.semilogy(range(0, len(data)), data)
    ax.set_title(title)
    fig.savefig(name)


def plotHeatMap(
        data: np.ndarray, title: str = "default",
        name: str = "default"):
    """[summary]
    Args:
        p (np.ndarray): [description]. 2-D matrix
    """
    printError(data.ndim != 2, "invalid dim")
    from matplotlib import cm

    fig, ax = plt.subplots()
    img = ax.imshow(data, cmap=cm.Reds)
    fig.colorbar(img)

    len_y, len_x = data.shape
    ax.set_xticks(np.arange(len_x))
    ax.set_yticks(np.arange(len_y))

    ax.set_title(title)
    fig.savefig(name)


def plotSparseMatrix(
        data: np.ndarray, title: str = "default",
        name: str = "default"):
    """[summary]
    Args:
        p (np.ndarray): [description]. 2-D matrix
    """
    printError(data.ndim != 2, "invalid dim")

    fig, ax = plt.subplots()
    ax.spy(data)
    ax.set_title(title)
    fig.savefig(name)
# <<< plot utils
