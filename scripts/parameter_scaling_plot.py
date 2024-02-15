import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.optimize import curve_fit

# color_list = ["purple", "#306B37", "darkgoldenrod", "#3F7BB6", "#BF4145", "#CF630A"]
# color_list = ["#2E4854", "#557B82", "#BAB2A9", "#C98769", "#A1553A", "darkgoldenrod"]
# color_list = ["#12464F", "#DF543E", "#C8943B", "#378F8C", "#AD9487"]
# color_list = ["#909B55", "#4B3025", "#AF5D25", "#C3B426", "#C8A895"]
color_list = ["#4D4D42", "#A0071F", "#2E6C9D", "#5F2E06", "#DF9906"]
ls_list = ["-", "--", "-.", ":", "-", "--"]


def rolling_average(data, window_size):
    """Calculate the rolling average of a list."""
    return [
        sum(data[i : i + window_size]) / window_size
        for i in range(len(data) - window_size + 1)
    ]


def plot_loss_small_models():
    parameter_count = 8672
    string_list = ["Gaussian", "Gaussian_fixed_var"]
    for i, s in enumerate(string_list):
        plt.figure(figsize=(8, 6))
        file_name = f"results/transformer_{parameter_count}_{s}_training.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        plt.plot(
            model_dict["train_epochs"],
            model_dict["train_losses"],
            color=color_list[i],
            ls=ls_list[i],
            alpha=0.2,
            zorder=-10,
        )
        plt.plot(
            model_dict["test_epochs"],
            model_dict["test_losses"],
            color=color_list[i],
            ls=ls_list[i],
            label=f"{parameter_count} parameters",
        )

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylim(-2, 10)
        plt.ylabel("Loss")
        plt.savefig(f"plots/loss_{parameter_count}_{s}.pdf", bbox_inches="tight")


def plot_loss():
    parameter_count_list = [8762]  # , 18418, 40418, 133218]
    for i, parameter_count in enumerate(parameter_count_list):
        plt.figure(figsize=(8, 6))
        file_name = f"results/transformer_{parameter_count}_training.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        plt.plot(
            model_dict["train_epochs"],
            model_dict["train_losses"],
            color=color_list[i],
            ls=ls_list[i],
            alpha=0.2,
            zorder=-10,
        )
        plt.plot(
            model_dict["test_epochs"],
            model_dict["test_losses"],
            color=color_list[i],
            ls=ls_list[i],
            label=f"{parameter_count} parameters",
        )

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylim(-5, 5)
        plt.ylabel("Loss")
        plt.savefig(f"plots/loss_{parameter_count}.pdf", bbox_inches="tight")


def plot_combined_losses():
    parameter_count_list = [2762, 17954, 879490]
    plt.figure(figsize=(8, 6))

    for i, parameter_count in enumerate(parameter_count_list):
        file_name = f"results/transformer_{parameter_count}_Gaussian_training.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        plt.plot(
            model_dict["train_epochs"],
            model_dict["train_losses"],
            color=color_list[i],
            ls=ls_list[i],
            alpha=0.2,
            zorder=-10,
        )
        # # Set your desired window size for the rolling average
        # window_size = 20  # You can adjust this value

        # # Calculate rolling average
        # rolled_avg_losses = rolling_average(
        #     np.clip(model_dict["test_losses"], -100, 1), window_size
        # )

        # # Adjust the epochs to match the length of the rolled average array
        # # (since the first few values don't have a full window)
        # rolled_avg_epochs = model_dict["test_epochs"][window_size - 1 :]

        plt.plot(
            model_dict["test_epochs"],
            model_dict["test_losses"],
            # rolled_avg_epochs,
            # rolled_avg_losses,
            color=color_list[i],
            ls=ls_list[i],
            label=f"{parameter_count} parameters",
        )

    plt.legend()
    plt.xlabel("Epoch")
    # plt.ylim(-5, 5)
    plt.ylabel("Loss")
    plt.savefig("plots/loss_Gaussian.pdf", bbox_inches="tight")


def parameter_test_scaling_plot():
    min_test_loss = []
    parameter_count_list = [2762, 17954, 879490]

    for parameter_count in parameter_count_list:
        file_name = f"results/transformer_{parameter_count}_Gaussian_training.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        min_test_loss.append(abs(min(model_dict["MSE_test_losses"])))

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

    # Extracting the parameters
    a, b = params
    print("Normalization constant: ", a)
    print("Power law exponent: ", b)

    N_arr = np.geomspace(1e3, 1e7)

    plt.figure(figsize=(8, 6))
    plt.scatter(parameter_count_list, min_test_loss, color=color_list[0])
    plt.plot(N_arr, power_law(N_arr, a, b), ls="--", color="gray")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of parameters")
    plt.ylabel("Minimum Test Loss")
    # plt.ylim(9e-2, 1.8e-1)
    # plt.xlim(5e3, 1e7)
    plt.savefig("plots/parameters_vs_loss_Gaussian.pdf", bbox_inches="tight")


def parameter_train_scaling_plot():
    min_test_loss = []
    parameter_count_list = [20641, 48961, 55458, 141473]

    for parameter_count in parameter_count_list:
        file_name = f"results/transformer_{parameter_count}_MSE_training.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        min_test_loss.append(abs(min(model_dict["train_losses"])))

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

    # Extracting the parameters
    a, b = params
    print("Normalization constant: ", a)
    print("Power law exponent: ", b)

    N_arr = np.geomspace(1e3, 1e7)

    plt.figure(figsize=(8, 6))
    plt.scatter(parameter_count_list, min_test_loss, color=color_list[0])
    plt.plot(N_arr, power_law(N_arr, a, b), ls="--", color="gray")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of parameters")
    plt.ylabel("Minimum train Loss")
    # plt.ylim(9e0, 1e1)
    plt.xlim(1e3, 1e7)
    plt.savefig(
        "plots/parameters_vs_train_loss_MSE_electricitytraffic.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    # plot_loss_small_models()
    parameter_test_scaling_plot()
    # parameter_train_scaling_plot()
    # plot_loss()
    plot_combined_losses()
