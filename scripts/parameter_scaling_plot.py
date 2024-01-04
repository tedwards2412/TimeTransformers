import matplotlib.pyplot as plt
import json


def plot_loss():
    file_name = f"results/transformer_{4850}_training.json"

    with open(file_name, "r") as file:
        model_dict = json.load(file)

    plt.figure(figsize=(8, 6))
    plt.plot(model_dict["train_epochs"], model_dict["train_losses"], label="Train loss")
    plt.plot(model_dict["test_epochs"], model_dict["test_losses"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.pdf", bbox_inches="tight")


def parameter_scaling_plot():
    parameter_count_list = [3466, 4850]
    min_test_loss = []

    for parameter_count in parameter_count_list:
        file_name = f"results/transformer_{parameter_count}_training.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        min_test_loss.append(min(model_dict["test_losses"]) + 10)

    plt.figure(figsize=(8, 6))
    plt.loglog(parameter_count_list, min_test_loss)
    plt.xlabel("Number of parameters")
    plt.ylabel("Loss")
    plt.savefig("parameters_vs_loss.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parameter_scaling_plot()
