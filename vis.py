#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt


def plot_categorical_distribution(probs, ax: plt.Axes = None,
                                  cla: bool = True,
                                  steps: int = None):
    """Plot box-plot with standard deviation of categorical distribution."""
    if ax is None:
        ax = plt.gca()
    if cla:
        ax.cla()

    if len(probs) <= 0:
        return

    num_actions = len(probs[0])
    ax.bar(
        np.arange(num_actions),
        np.mean(probs, axis=0),
        yerr=np.std(probs, axis=0),
        alpha=0.8)
    ax.set_title(F'Action probability distribution | {steps} steps')
    ax.set_xlabel('action')
    ax.set_ylabel('probability')
    ax.grid()
    plt.savefig('/tmp/actions.png')


def obs2vis(obs: np.ndarray):
    return (obs + 0.5).reshape((7, 7, 3))[..., ::-1]
