import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl

# color_list = ["purple", "#306B37", "darkgoldenrod", "#3F7BB6", "#BF4145", "#CF630A"]
# color_list = ["#2E4854", "#557B82", "#BAB2A9", "#C98769", "#A1553A", "darkgoldenrod"]
# color_list = ["#12464F", "#DF543E", "#C8943B", "#378F8C", "#AD9487"]
# color_list = ["#909B55", "#4B3025", "#AF5D25", "#C3B426", "#C8A895"]
params = {
    "font.size": 18,
    "legend.fontsize": 18,
    "legend.frameon": False,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.figsize": (7, 5),
    "xtick.top": True,
    "axes.unicode_minus": False,
    "ytick.right": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.major.pad": 8,
    "xtick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.major.size": 8,
    "ytick.minor.size": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.5,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": "cmr10",
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # needed when using cm=cmr10 for normal text
}
mpl.rcParams.update(params)

color_list = [
    "#4D4D42",
    "#A0071F",
    "#2E6C9D",
    "#5F2E06",
    "#DF9906",
    "#12464F",
    "#AF5D25",
]
ls_list = ["-", "--", "-.", ":", "-", "--", "-."]


def rolling_average(data, window_size):
    """Calculate the rolling average of a list."""
    return [
        sum(data[i : i + window_size]) / window_size
        for i in range(len(data) - window_size + 1)
    ]


def plot_combined_losses():
    # parameter_count_list = [2771, 8387, 24451, 130051, 879619, 3430403, 5013507]
    # parameter_count_list = [2771, 24451, 3430403, 130051, 19857411]
    parameter_count_list = [155011, 3825155, 21433347]
    fig, axes = plt.subplots(1, 3, figsize=(30, 6))

    for i, parameter_count in enumerate(parameter_count_list):
        file_name = f"results/parameterscaling_{parameter_count}_studentT_training.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        axes[0].plot(
            model_dict["train_epochs"],
            model_dict["train_losses"],
            color=color_list[i],
            ls=ls_list[i],
            alpha=0.2,
            zorder=-10,
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Train Loss")

        axes[1].plot(
            model_dict["test_epochs"],
            model_dict["test_losses"],
            color=color_list[i],
            ls=ls_list[i],
            label=f"{parameter_count} parameters",
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Test Loss")

        axes[2].semilogy(
            model_dict["test_epochs"],
            model_dict["CRPS_test_losses"],
            color=color_list[i],
            ls=ls_list[i],
            label=f"{parameter_count} parameters",
        )
        axes[2].legend(ncol=2, loc="upper right")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("CRPS Test Loss")

    plt.savefig("plots/loss_studentT.pdf", bbox_inches="tight")


def parameter_MSE_CRPS_scaling_plot():
    min_MSE_test_loss = []
    min_CRPS_test_loss = []
    min_test_loss = []
    # parameter_count_list = [2771, 8387, 24451, 130051, 879619, 3430403, 5013507]
    # parameter_count_list = [2771, 24451, 3430403, 130051, 19857411]
    parameter_count_list = [155011, 3825155, 21433347]
    # parameter_count_list = [2771, 24451, 130051, 3430403, 19857411]
    # 130051, 24451, 2771, 5013507, 8387, 879619

    for parameter_count in parameter_count_list:
        file_name = f"results/parameterscaling_{parameter_count}_studentT_training.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        min_MSE_test_loss.append(min(model_dict["MSE_test_losses"]))
        min_CRPS_test_loss.append(min(model_dict["CRPS_test_losses"]))
        min_test_loss.append(min(model_dict["test_losses"]) + 1)

    min_MSE_test_loss = np.array(min_MSE_test_loss)
    min_CRPS_test_loss = np.array(min_CRPS_test_loss)
    min_test_loss = np.array(min_test_loss)
    parameter_count_list = np.array(parameter_count_list)

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
    print("Test Loss + 1")
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

    N_arr = np.geomspace(1e3, 1e8)

    fig, axes = plt.subplots(1, 3, figsize=(30, 6))
    axes[0].scatter(parameter_count_list, min_MSE_test_loss, color=color_list[0])
    axes[0].loglog(N_arr, power_law(N_arr, a_MSE, b_MSE), ls="--", color="gray")
    axes[0].set_xlabel("Number of parameters")
    axes[0].set_ylabel("Minimum MSE Test Loss")

    axes[1].scatter(parameter_count_list, min_CRPS_test_loss, color=color_list[0])
    axes[1].loglog(N_arr, power_law(N_arr, a_CRPS, b_CRPS), ls="--", color="gray")
    axes[1].set_xlabel("Number of parameters")
    axes[1].set_ylabel("Minimum CRPS Test Loss")

    axes[2].scatter(parameter_count_list, min_test_loss, color=color_list[0])
    axes[2].loglog(N_arr, power_law(N_arr, a, b), ls="--", color="gray")
    axes[2].set_xlabel("Number of parameters")
    axes[2].set_ylabel("Minimum Test Loss + 1")

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


if __name__ == "__main__":
    parameter_MSE_CRPS_scaling_plot()
    # parameter_test_scaling_plot()
    plot_combined_losses()
