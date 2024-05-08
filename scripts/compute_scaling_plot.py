import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl

import os
import torch
from matplotlib.ticker import NullLocator
import yaml
import glob

# This is just until temporary implementation
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

import Transformer

plt.style.use("plots.mplstyle")


def rolling_average(data, window_size):
    """Calculate the rolling average of a list."""
    return [
        sum(data[i : i + window_size]) / window_size
        for i in range(len(data) - window_size + 1)
    ]


def compute_losses_scaling_plot():
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 3),
    )
    plt.subplots_adjust(wspace=0.16)
    C = 2
    B = 512

    json_files = glob.glob("results/parameterscaling*.json")
    for json_file in json_files[:2]:
        with open(json_file, "r") as file:
            data = json.load(file)
        axes[0].loglog(6 * B * data["Nparams"] * data["test_epochs"], data["MSE_test_losses"], c='k', alpha=0.1)
        axes[1].loglog(6 * B * data["Nparams"] * data["test_epochs"], data["CRPS_test_losses"], c='k', alpha=0.1)
        axes[2].loglog(6 * B * data["Nparams"] * data["test_epochs"], data["test_losses"] + C, c='k', alpha=0.1)

    json_files = glob.glob("results/maxLRscaling*.json")
    for json_file in json_files[:2]:
        with open(json_file, "r") as file:
            data = json.load(file)
        axes[0].loglog(6 * B * data["Nparams"] * data["test_epochs"], data["MSE_test_losses"], c='k', alpha=0.1)
        axes[1].loglog(6 * B * data["Nparams"] * data["test_epochs"], data["CRPS_test_losses"], c='k', alpha=0.1)
        axes[2].loglog(6 * B * data["Nparams"] * data["test_epochs"], data["test_losses"] + C, c='k', alpha=0.1)

    plt.savefig("plots/compute_scaling_plot.pdf", bbox_inches="tight")


if __name__ == "__main__":
    compute_losses_scaling_plot()
    # parameter_test_scaling_plot()
    # plot_combined_losses()
    # test_parameter_dist()
