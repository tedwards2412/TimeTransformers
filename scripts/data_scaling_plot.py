import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.optimize import curve_fit

plt.style.use("plots.mplstyle")


def data_MSEtest_scaling_plot():
    min_test_loss = []
    train_tokens = []
    nparams = 5013507
    token_count_list = [2771, 8387, 24451, 130051, 879619, 3430403, 5013507]

    for tokens in token_count_list:
        file_name = f"results/transformer_{nparams}_studentT_{tokens}_datascaling.json"

        with open(file_name, "r") as file:
            model_dict = json.load(file)

        min_test_loss.append(min(model_dict["MSE_test_losses"]))
        train_tokens.append(model_dict["Ntrain_tokens"])

    min_test_loss = np.array(min_test_loss)
    train_tokens = np.array(train_tokens)

    # Power law function
    # def power_law(x, loga, b):
    #     a = 10**loga
    #     return (x / a) ** b

    # Curve fitting
    # params, covariance = curve_fit(
    #     power_law, train_tokens, min_test_loss, p0=[5.0, -1.0]
    # )

    # Extracting the parameters
    # a, b = params
    # print("Normalization constant: ", a)
    # print("Power law exponent: ", b)

    # N_arr = np.geomspace(1e3, 1e7)

    plt.figure(figsize=(8, 6))
    plt.scatter(train_tokens, min_test_loss, color=color_list[0])
    # plt.plot(N_arr, power_law(N_arr, a, b), ls="--", color="gray")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of training tokens")
    plt.ylabel("Minimum MSE Test Loss")
    # plt.ylim(9e-2, 1.8e-1)
    # plt.xlim(5e3, 1e7)
    plt.savefig("plots/data_vs_loss_studentT.pdf", bbox_inches="tight")


# def data_test_scaling_plot():
#     min_test_loss = []
#     parameter_count_list = [2771, 8387, 24451, 130051, 879619, 3430403, 5013507]

#     for parameter_count in parameter_count_list:
#         file_name = f"results/transformer_{parameter_count}_studentT_training.json"

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

#     N_arr = np.geomspace(1e3, 1e7)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(parameter_count_list, min_test_loss, color=color_list[0])
#     plt.plot(N_arr, power_law(N_arr, a, b), ls="--", color="gray")
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.xlabel("Number of parameters")
#     plt.ylabel("Minimum Test Loss + 1")
#     # plt.ylim(9e0, 1e1)
#     plt.xlim(1e3, 1e7)
#     plt.savefig("plots/data_vs_testloss_studentT.pdf", bbox_inches="tight")


if __name__ == "__main__":
    data_MSEtest_scaling_plot()
    # data_test_scaling_plot()
