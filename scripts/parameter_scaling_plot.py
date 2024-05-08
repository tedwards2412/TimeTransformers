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


def parameter_losses_scaling_plot():
    min_MSE_test_loss = []
    min_CRPS_test_loss = []
    min_test_loss = []
    parameter_count_list = []
    C = 2

    json_files = glob.glob("results/parameterscaling*.json")
    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)

        min_MSE_test_loss.append(min(data["MSE_test_losses"]))
        min_CRPS_test_loss.append(min(data["CRPS_test_losses"]))
        min_test_loss.append(min(data["test_losses"]) + C)
        parameter_count_list.append(data["Nparams"])

    json_files = glob.glob("results/maxLRscaling*.json")
    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)

        min_MSE_test_loss.append(min(data["MSE_test_losses"]))
        min_CRPS_test_loss.append(min(data["CRPS_test_losses"]))
        min_test_loss.append(min(data["test_losses"]) + C)
        parameter_count_list.append(data["Nparams"])

    min_MSE_test_loss = np.array(min_MSE_test_loss)
    min_CRPS_test_loss = np.array(min_CRPS_test_loss)
    min_test_loss = np.array(min_test_loss)
    parameter_count_list = np.array(parameter_count_list)

    # Find the best model, i.e. remove duplicates
    unique_models = np.unique(parameter_count_list)

    for model in unique_models:
        idx = np.where(parameter_count_list == model)
        min_MSE_test_loss[idx] = min(min_MSE_test_loss[idx])
        min_CRPS_test_loss[idx] = min(min_CRPS_test_loss[idx])
        min_test_loss[idx] = min(min_test_loss[idx])

    # Power law function
    def power_law(x, loga, b):
        a = 10**loga
        return (x / a) ** b

    # Curve fitting
    params, covariance = curve_fit(
        power_law, parameter_count_list, min_test_loss, p0=[5.0, -1.0]
    )

    params_MSE, covariance_MSE = curve_fit(
        power_law, parameter_count_list, min_MSE_test_loss, p0=[5.0, -1.0]
    )

    params_CRPS, covariance_CRPS = curve_fit(
        power_law, parameter_count_list, min_CRPS_test_loss, p0=[5.0, -1.0]
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

    N_arr = np.geomspace(1e3, 1e9)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 3),
    )
    plt.subplots_adjust(wspace=0.16)
    axes[0].scatter(parameter_count_list, min_MSE_test_loss, color="C2", s=100)
    axes[0].loglog(N_arr, power_law(N_arr, a_MSE, b_MSE), ls="--", color="C0")
    # axes[0].set_xlabel("Number of parameters")
    axes[0].set_ylabel("In Sequence Test Loss")
    axes[0].set_title("MSE", fontweight="bold")
    axes[0].set_xlim(1e3, 1e9)
    axes[0].set_ylim(0.06, 0.16)
    axes[0].set_yticks([0.06, 0.08, 0.1, 0.12, 0.14, 0.16], minor=False)
    axes[0].set_yticklabels(["0.06", "0.08", "0.10", "0.12", "0.14", "0.16"])

    axes[1].scatter(parameter_count_list, min_CRPS_test_loss, color="C2", s=100)
    axes[1].loglog(N_arr, power_law(N_arr, a_CRPS, b_CRPS), ls="--", color="C0")
    axes[1].set_title(f"CRPS")
    axes[1].set_yticks([0.06, 0.08, 0.1, 0.12, 0.14, 0.16], minor=False)
    axes[1].set_yticklabels(["0.06", "0.08", "0.10", "0.12", "0.14", "0.16"])
    axes[1].set_ylim(0.06, 0.16)

    axes[1].set_xlim(1e3, 1e9)
    # axes[1].set_ylim(0.04, 0.16)

    axes[2].scatter(parameter_count_list, min_test_loss, color="C2", s=100)
    axes[2].loglog(N_arr, power_law(N_arr, a, b), ls="--", color="C0")
    # axes[2].set_xlabel("Number of parameters")
    # axes[2].set_ylabel("Minimum Test Loss + 2")
    axes[2].set_title(f"Log Likelihood (+2)")
    axes[2].yaxis.set_minor_locator(NullLocator())
    axes[2].set_yticks([0.2, 0.4, 0.6, 0.8, 1.2, 1.6], minor=False)
    axes[2].set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.2", "1.6"])
    axes[2].set_ylim(0.2, 1.6)
    axes[2].set_xlim(1e3, 1e9)

    # Shared xlabel
    fig.text(0.5, -0.04, "Number of parameters", ha="center", va="center", fontsize=20)

    plt.savefig("plots/parameters_vs_loss_studentT.pdf", bbox_inches="tight")


# def parameter_test_scaling_plot():
#     min_test_loss = []
#     # parameter_count_list = [2771, 8387, 24451, 130051, 879619, 3430403, 5013507]
#     # parameter_count_list = [2771, 24451, 5013507, 19857411]
#     parameter_count_list = [2771, 130051, 19857411]

#     for parameter_count in parameter_count_list:
#         file_name = f"results/parameterscaling_{parameter_count}_studentT_training.json"

#         with open(file_name, "r") as file:
#             model_dict = json.load(file)

#         min_test_loss.append(min(model_dict["test_losses"]) + 1)

#     min_test_loss = np.array(min_test_loss)
#     parameter_count_list = np.array(parameter_count_list)

#     # Power law function
#     def power_law(x, loga, b):
#         a = 10**loga
#         return (x / a) ** b

#     # Curve fitting
#     params, covariance = curve_fit(
#         power_law, parameter_count_list, min_test_loss, p0=[5.0, -0.0]
#     )

#     # Extracting the parameters
#     a, b = params
#     print("Normalization constant: ", a)
#     print("Power law exponent: ", b)

#     N_arr = np.geomspace(1e3, 1e8)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(parameter_count_list, min_test_loss, color=color_list[0])
#     plt.plot(N_arr, power_law(N_arr, a, b), ls="--", color="gray")
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.xlabel("Number of parameters")
#     plt.ylabel("Minimum Test Loss + 1")
#     # plt.ylim(9e0, 1e1)
#     plt.xlim(1e3, 5e7)
#     plt.savefig("plots/parameters_vs_testloss_studentT.pdf", bbox_inches="tight")


def train(config):
    with open(config, "r") as file:
        config = yaml.safe_load(file)
    # Transformer parameters
    output_dim = config["transformer"]["output_dim"]
    d_model = config["transformer"]["d_model"]
    num_heads = config["transformer"]["num_heads"]
    num_layers = config["transformer"]["num_layers"]
    d_ff = config["transformer"]["d_ff"]
    dropout = config["transformer"]["dropout"]
    num_distribution_layers = config["transformer"]["num_distribution_layers"]
    max_seq_length = config["train"]["max_seq_length"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available() else "cpu"
        else "cpu"
    )
    print(f"Using {device}")

    transformer = Transformer.Decoder_Transformer(
        output_dim,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        num_distribution_layers,
        # patch_size,
        device=device,
    ).to(device)
    num_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Number of parameters: ", num_params)
    print("Aspect ratio: ", d_model / num_layers)
    print("Number of heads: ", num_heads)

    return num_params, num_heads, d_model / num_layers


def test_parameter_dist():
    yml_files = glob.glob("configs/parameter_scaling/*.yml")
    N_list = []
    nheads_list = []
    CRPS_min = []
    AR_list = []

    for config_file in yml_files:
        print(config_file)
        N, nh, AR = train(config_file)
        N_list.append(N)
        nheads_list.append(nh)
        CRPS_min.append(0.1)
        AR_list.append(AR)

    plt.scatter(N_list, AR_list)
    plt.xlim(1e3, 1e9)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of parameters")
    plt.ylabel("Aspect ratio")
    # plt.show()
    plt.savefig("plots/parameter_scaling_dist.pdf", bbox_inches="tight")

    return None


if __name__ == "__main__":
    parameter_losses_scaling_plot()
    # parameter_test_scaling_plot()
    # plot_combined_losses()
    # test_parameter_dist()
