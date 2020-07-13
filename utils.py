# -*- coding: utf-8 -*-
"""utils.py

Author -- Mathias Wöß
Contact -- k11709064@students.jku.at
"""

import os
from matplotlib import pyplot as plt
from typing import List
import numpy as np


def de_normalize(data, means: list, stds: list, reshape: list = None) -> List[np.ndarray]:
    """De-normalize a list of data according to a list of means and stds, optionally reshape input first"""
    # (N, n_channels, X, Y) -- normalized
    input_list = []
    for i, sample in enumerate(data):  # (n_channels, X, Y)
        if reshape:  # tensor(n_channels, X*Y) ->  (n_channels, X, Y)
            sample = sample.detach().numpy().reshape(reshape[i])
        # (X, Y)
        sample *= stds[i]
        sample += means[i]
        sample = sample.astype(np.uint8)
        input_list.append(sample)
    return input_list


def plot(inputs, targets, predictions, means, stds, path, update):
    """Plotting the inputs, targets, predictions and their difference to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(2, 2)
    plot_inputs = de_normalize(inputs[:, 0, ...], means, stds)
    plot_predictions = de_normalize(predictions, means, stds, reshape=[t.shape for t in targets])
    plot_targets = de_normalize(targets, means, stds)
    plot_difference = [np.abs(t - p) for t, p in zip(plot_targets, plot_predictions)]

    for i in range(len(inputs)):
        ax[0, 0].clear()
        ax[0, 0].set_title('input')
        ax[0, 0].imshow(plot_inputs[i], cmap=plt.cm.gray, interpolation='none')
        ax[0, 0].set_axis_off()
        ax[0, 1].clear()
        ax[0, 1].set_title('targets')
        ax[0, 1].imshow(plot_targets[i], cmap=plt.cm.gray, interpolation='none')
        ax[0, 1].set_axis_off()
        ax[1, 0].clear()
        ax[1, 0].set_title('predictions')
        ax[1, 0].imshow(plot_predictions[i], cmap=plt.cm.gray, interpolation='none')
        ax[1, 0].set_axis_off()
        ax[1, 1].clear()
        ax[1, 1].set_title('difference')
        ax[1, 1].imshow(plot_difference[i], interpolation='none')
        ax[1, 1].set_axis_off()
        # fig.tight_layout()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=850)
    del fig
