import matplotlib.pyplot as plt
import json

# color_list = ["purple", "#306B37", "darkgoldenrod", "#3F7BB6", "#BF4145", "#CF630A"]
color_list = ["#2E4854", "#557B82", "#BAB2A9", "#C98769", "#A1553A", "darkgoldenrod"]
ls_list = ["-", "--", "-.", ":", "-", "--"]


def plot_loss():
    parameter_count_list = [1198, 2922, 4850, 7426, 20146, 33570]  # , 165826]

    plt.figure(figsize=(8, 6))

    for i, parameter_count in enumerate(parameter_count_list):
        file_name = f"results/transformer_{parameter_count}_training.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        plt.plot(
            model_dict["train_epochs"],
            model_dict["train_losses"],
            color=color_list[i],
            ls=ls_list[i],
            alpha=0.2,
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
    plt.ylim(-10, 0)
    plt.ylabel("Loss")
    plt.savefig("plots/loss.pdf", bbox_inches="tight")


def parameter_scaling_plot():
    parameter_count_list = [1198, 2922, 4850, 7426, 20146, 33570]  # , 165826]
    min_test_loss = []

    for parameter_count in parameter_count_list:
        file_name = f"results/transformer_{parameter_count}_training.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        min_test_loss.append(min(model_dict["test_losses"]) + 10)

    plt.figure(figsize=(8, 6))
    plt.scatter(parameter_count_list, min_test_loss, color=color_list[0])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of parameters")
    plt.ylabel("Minimum Test Loss")
    plt.savefig("plots/parameters_vs_loss.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parameter_scaling_plot()
    plot_loss()
