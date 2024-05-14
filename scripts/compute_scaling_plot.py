import matplotlib.pyplot as plt
import json
import numpy as np
import scipy.ndimage
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


def exponential_smoothing(data, alpha):
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # Initialize with the first data point
    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t - 1]
    return smoothed_data


def compute_losses_scaling_plot():
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 3),
    )
    plt.subplots_adjust(wspace=0.16)
    C = 2
    B = 512

    interp_arr = np.geomspace(1e-9, 3.2e-4, 200)
    min_mse_arr = np.ones_like(interp_arr) * 10.0
    min_crps_arr = np.ones_like(interp_arr) * 10.0
    min_test_loss_arr = np.ones_like(interp_arr) * 10.0
    json_files = glob.glob("results/parameterscaling*.json")
    for json_file in json_files:
        print(json_file)
        with open(json_file, "r") as file:
            data = json.load(file)
        mask = interp_arr <= 6 * B * data["Nparams"] * np.array(data["test_epochs"])[
            -1
        ] / (8.64 * 1e19)
        mse_interp = np.interp(
            interp_arr[mask],
            6 * B * data["Nparams"] * np.array(data["test_epochs"]) / (8.64 * 1e19),
            np.array(data["MSE_test_losses"]),
        )
        crps_interp = np.interp(
            interp_arr[mask],
            6 * B * data["Nparams"] * np.array(data["test_epochs"]) / (8.64 * 1e19),
            np.array(data["CRPS_test_losses"]),
        )
        test_loss_interp = np.interp(
            interp_arr[mask],
            6 * B * data["Nparams"] * np.array(data["test_epochs"]) / (8.64 * 1e19),
            np.array(data["test_losses"]) + C,
        )
        min_mse_arr[mask] = np.minimum(min_mse_arr[mask], mse_interp)
        min_crps_arr[mask] = np.minimum(min_crps_arr[mask], crps_interp)
        min_test_loss_arr[mask] = np.minimum(min_test_loss_arr[mask], test_loss_interp)
        mse_mask = np.array(data["MSE_test_losses"]) < 0.5
        axes[0].loglog(
            6
            * B
            * data["Nparams"]
            * np.array(data["test_epochs"])[mse_mask]
            / (8.64 * 1e19),
            np.array(data["MSE_test_losses"])[mse_mask],
            c="k",
            alpha=0.1,
            lw=0.1,
        )
        axes[1].loglog(
            6 * B * data["Nparams"] * np.array(data["test_epochs"]) / (8.64 * 1e19),
            np.array(data["CRPS_test_losses"]),
            c="k",
            alpha=0.1,
            lw=0.1,
        )
        axes[2].loglog(
            6 * B * data["Nparams"] * np.array(data["test_epochs"]) / (8.64 * 1e19),
            np.array(data["test_losses"]) + C,
            c="k",
            alpha=0.1,
            lw=0.1,
        )

    json_files = glob.glob("results/maxLRscaling*.json")
    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)
        mask = interp_arr <= 6 * B * data["Nparams"] * np.array(data["test_epochs"])[
            -1
        ] / (8.64 * 1e19)
        mse_interp = np.interp(
            interp_arr[mask],
            6 * B * data["Nparams"] * np.array(data["test_epochs"]) / (8.64 * 1e19),
            np.array(data["MSE_test_losses"]),
        )
        crps_interp = np.interp(
            interp_arr[mask],
            6 * B * data["Nparams"] * np.array(data["test_epochs"]) / (8.64 * 1e19),
            np.array(data["CRPS_test_losses"]),
        )
        test_loss_interp = np.interp(
            interp_arr[mask],
            6 * B * data["Nparams"] * np.array(data["test_epochs"]) / (8.64 * 1e19),
            np.array(data["test_losses"]) + C,
        )
        min_mse_arr[mask] = np.minimum(min_mse_arr[mask], mse_interp)
        min_crps_arr[mask] = np.minimum(min_crps_arr[mask], crps_interp)
        min_test_loss_arr[mask] = np.minimum(min_test_loss_arr[mask], test_loss_interp)
        mse_mask = np.array(data["MSE_test_losses"]) < 0.5
        axes[0].loglog(
            6
            * B
            * data["Nparams"]
            * np.array(data["test_epochs"])[mse_mask]
            / (8.64 * 1e19),
            np.array(data["MSE_test_losses"])[mse_mask],
            c="k",
            alpha=0.1,
            lw=0.1,
        )
        axes[1].loglog(
            6 * B * data["Nparams"] * np.array(data["test_epochs"]) / (8.64 * 1e19),
            np.array(data["CRPS_test_losses"]),
            c="k",
            alpha=0.1,
            lw=0.1,
        )
        axes[2].loglog(
            6 * B * data["Nparams"] * np.array(data["test_epochs"]) / (8.64 * 1e19),
            np.array(data["test_losses"]) + C,
            c="k",
            alpha=0.1,
            lw=0.1,
        )
    from scipy.ndimage import gaussian_filter1d

    axes[0].loglog(
        interp_arr,
        min_mse_arr,
        # gaussian_filter1d(min_mse_arr, sigma=10),
        c="C2",
        lw=2,
        label="Minimum MSE",
    )
    axes[0].loglog(
        np.geomspace(1e-7, 1e-3, 100),
        0.09 * (np.geomspace(1e-7, 1e-3, 100) / 4e-7) ** (-0.025),
        # gaussian_filter1d(min_mse_arr, sigma=10),
        c="C1",
        lw=2,
        ls=(1, (5, 1)),
        label="Minimum MSE",
    )
    axes[1].loglog(
        interp_arr,
        min_crps_arr,
        # gaussian_filter1d(min_crps_arr, sigma=10),
        c="C2",
        lw=2,
        label="Minimum CRPS",
    )
    axes[1].loglog(
        np.geomspace(1e-7, 1e-3, 100),
        0.0925 * (np.geomspace(1e-7, 1e-3, 100) / 4e-7) ** (-0.025),
        # gaussian_filter1d(min_mse_arr, sigma=10),
        c="C1",
        lw=2,
        ls=(1, (5, 1)),
        label="Minimum MSE",
    )
    axes[2].loglog(
        interp_arr,
        min_test_loss_arr,
        # exponential_smoothing(min_test_loss_arr, alpha=0.14),
        c="C2",
        lw=2,
        label="Minimum Test Loss",
    )
    axes[2].loglog(
        np.geomspace(1e-9, 1e-3, 100),
        0.77 * (np.geomspace(1e-9, 1e-3, 100) / 4e-7) ** (-0.12),
        # gaussian_filter1d(min_mse_arr, sigma=10),
        c="C1",
        lw=2,
        ls=(1, (5, 1)),
        label="Minimum MSE",
    )
    for i in range(3):
        axes[i].set_xlim(1e-9, 1e-3)
    axes[0].set_ylim(7e-2, 0.2)
    axes[1].set_ylim(7e-2, 0.2)
    axes[2].set_ylim(0.3, 2.0)
    axes[0].set_ylabel("In-sequence Test Loss")
    axes[0].set_title("MSE", fontweight="bold")
    axes[1].set_title(f"CRPS")
    axes[2].set_title(f"Log-likelihood (+2)")
    for i in range(3):
        axes[i].set_xticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3], minor=False)
    axes[0].set_yticks([0.06, 0.08, 0.1, 0.12, 0.14, 0.16], minor=False)
    axes[0].set_yticklabels(["0.06", "0.08", "0.10", "0.12", "0.14", "0.16"])
    axes[1].set_yticks([0.06, 0.08, 0.1, 0.12, 0.14, 0.16], minor=False)
    axes[1].set_yticklabels(["0.06", "0.08", "0.10", "0.12", "0.14", "0.16"])
    axes[2].set_yticks([0.3, 0.4, 0.6, 0.8, 1.2, 1.6], minor=False)
    axes[2].set_yticklabels(["0.3", "0.4", "0.6", "0.8", "1.2", "1.6"])
    plt.subplots_adjust(wspace=0.16)
    fig.text(0.5, -0.07, "Compute [PF-days]", ha="center", va="center", fontsize=20)
    plt.savefig("plots/compute_scaling_plot.pdf", bbox_inches="tight")


if __name__ == "__main__":
    compute_losses_scaling_plot()
    # parameter_test_scaling_plot()
    # plot_combined_losses()
    # test_parameter_dist()
