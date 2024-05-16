import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib.ticker import NullLocator

import os
import torch
import yaml
import glob

# This is just until temporary implementation
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

import Transformer

plt.style.use("plots.mplstyle")

# def data_MSEtest_scaling_plot():
#     min_test_loss = []
#     train_tokens = []
#     nparams = 5013507
#     token_count_list = [2771, 8387, 24451, 130051, 879619, 3430403, 5013507]

#     for tokens in token_count_list:
#         file_name = f"results/transformer_{nparams}_studentT_{tokens}_datascaling.json"

#         with open(file_name, "r") as file:
#             model_dict = json.load(file)

#         min_test_loss.append(min(model_dict["MSE_test_losses"]))
#         train_tokens.append(model_dict["Ntrain_tokens"])

#     min_test_loss = np.array(min_test_loss)
#     train_tokens = np.array(train_tokens)

#     # Power law function
#     # def power_law(x, loga, b):
#     #     a = 10**loga
#     #     return (x / a) ** b

#     # Curve fitting
#     # params, covariance = curve_fit(
#     #     power_law, train_tokens, min_test_loss, p0=[5.0, -1.0]
#     # )

#     # Extracting the parameters
#     # a, b = params
#     # print("Normalization constant: ", a)
#     # print("Power law exponent: ", b)

#     # N_arr = np.geomspace(1e3, 1e7)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(train_tokens, min_test_loss, color=color_list[0])
#     # plt.plot(N_arr, power_law(N_arr, a, b), ls="--", color="gray")
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.xlabel("Number of training tokens")
#     plt.ylabel("Minimum MSE Test Loss")
#     # plt.ylim(9e-2, 1.8e-1)
#     # plt.xlim(5e3, 1e7)
#     plt.savefig("plots/data_vs_loss_studentT.pdf", bbox_inches="tight")


def parameter_losses_scaling_plot():
    min_MSE_test_loss = []
    min_CRPS_test_loss = []
    min_test_loss = []
    datasetsize_list = []

    json_files = glob.glob("results/datascaling*.json")
    final_file = "results/parameterscaling_21433347_studentT_training.json"
    json_files.append(final_file)

    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)

        min_MSE_test_loss.append(min(data["MSE_test_losses"]))
        min_CRPS_test_loss.append(min(data["CRPS_test_losses"]))
        min_test_loss.append(min(data["test_losses"]) + 3)
        try:
            datasetsize_list.append(data["Ntrain_tokens"] + data["Ntest_tokens"])
        except:
            datasetsize_list.append(7695618350 + 419926020)

    min_MSE_test_loss = np.array(min_MSE_test_loss)
    min_CRPS_test_loss = np.array(min_CRPS_test_loss)
    min_test_loss = np.array(min_test_loss)
    datasetsize_list = np.array(datasetsize_list)

    # Power law function
    def power_law(x, loga, b):
        a = 10**loga
        return (x / a) ** b

    # Curve fitting
    params, covariance = curve_fit(
        power_law, datasetsize_list, min_test_loss, p0=[5.0, -1.0]
    )

    params_MSE, covariance_MSE = curve_fit(
        power_law, datasetsize_list, min_MSE_test_loss, p0=[5.0, -1.0]
    )

    params_CRPS, covariance_CRPS = curve_fit(
        power_law, datasetsize_list, min_CRPS_test_loss, p0=[5.0, -1.0]
    )

    # Extracting the parameters
    a, b = params
    print("Test Loss + 2")
    print("Normalization constant: ", a)
    print("Power law exponent: ", b)

    print("MSE")
    a_MSE, b_MSE = params_MSE
    print("Normalization constant: ", a_MSE)
    print("Power law exponent: ", b_MSE)

    print("CRPS")
    a_CRPS, b_CRPS = params_CRPS
    print("Normalization constant: ", a_CRPS)
    print("Power law exponent: ", b_CRPS)

    N_arr = np.geomspace(1e5, 1e11)

    # fig, axes = plt.subplots(1, 3, figsize=(30, 6))
    # axes[0].scatter(datasetsize_list, min_MSE_test_loss)
    # axes[0].loglog(N_arr, power_law(N_arr, a_MSE, b_MSE), ls="--", color="gray")
    # axes[0].set_xlabel("Data set size")
    # axes[0].set_ylabel("Minimum MSE Test Loss")

    # axes[1].scatter(datasetsize_list, min_CRPS_test_loss)
    # axes[1].loglog(N_arr, power_law(N_arr, a_CRPS, b_CRPS), ls="--", color="gray")

    # axes[1].set_xlabel("Data set size")
    # axes[1].set_ylabel("Minimum CRPS Test Loss")

    # # axes[1].set_xlim(1e3, 1e11)
    # # axes[1].set_ylim(0.04, 0.16)

    # axes[2].scatter(datasetsize_list, min_test_loss)
    # axes[2].loglog(N_arr, power_law(N_arr, a, b), ls="--", color="gray")
    # axes[2].set_xlabel("Data set size")
    # axes[2].set_ylabel("Minimum Test Loss + 2")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 3),
    )
    plt.subplots_adjust(wspace=0.16)
    axes[0].scatter(datasetsize_list, min_MSE_test_loss, color="C2", s=100)
    axes[0].loglog(N_arr, power_law(N_arr, a_MSE, b_MSE), ls="--", color="C0")
    # axes[0].set_xlabel("Number of parameters")
    axes[0].set_ylabel("In Sequence Test Loss")
    axes[0].set_title("MSE", fontweight="bold")
    axes[0].set_xlim(1e5, 1e11)
    axes[0].set_ylim(0.05, 0.16)
    axes[0].set_yticks([0.06, 0.08, 0.1, 0.12, 0.14, 0.16], minor=False)
    axes[0].set_yticklabels(["0.06", "0.08", "0.10", "0.12", "0.14", "0.16"])

    axes[1].scatter(datasetsize_list, min_CRPS_test_loss, color="C2", s=100)
    axes[1].loglog(N_arr, power_law(N_arr, a_CRPS, b_CRPS), ls="--", color="C0")
    axes[1].set_title(f"CRPS")
    axes[1].set_yticks([0.06, 0.08, 0.1, 0.12, 0.14, 0.16], minor=False)
    axes[1].set_yticklabels(["0.06", "0.08", "0.10", "0.12", "0.14", "0.16"])
    axes[1].set_ylim(0.05, 0.16)
    axes[1].set_xlim(1e5, 1e11)

    axes[2].scatter(datasetsize_list, min_test_loss, color="C2", s=100)
    axes[2].loglog(N_arr, power_law(N_arr, a, b), ls="--", color="C0")
    axes[2].set_title(f"Log Likelihood (+3)")
    axes[2].yaxis.set_minor_locator(NullLocator())
    # axes[2].set_yticks([0.2, 0.4, 0.6, 0.8, 1.2, 1.6], minor=False)
    # axes[2].set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.2", "1.6"])
    # axes[2].set_ylim(0.2, 1.6)
    axes[2].set_xlim(1e5, 1e11)

    # Shared xlabel
    fig.text(
        0.5, -0.06, "Number of Tokens in Dataset", ha="center", va="center", fontsize=20
    )

    plt.savefig("plots/data_vs_loss_studentT.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parameter_losses_scaling_plot()
    # data_test_scaling_plot()
