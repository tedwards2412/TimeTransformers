import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pickle


def fit_max_LR_func():
    parameter_count_list = np.array([104579, 580355, 2242051, 18277379])
    highest_LR = np.array([0.05, 0.0018803, 0.00070711, 0.00070711])

    # Power law function
    def power_law(x, loga, b):
        a = 10**loga
        return (x / a) ** b

    def other_func(x, loga, b, c):
        a = 10**loga
        return c + (x / a) ** b

    # Curve fitting
    params, covariance = curve_fit(
        power_law, parameter_count_list, highest_LR, p0=[5.0, -1.0]
    )
    params2, covariance2 = curve_fit(
        other_func,
        parameter_count_list,
        highest_LR,
        p0=[5.0, -1.0, 1e-3],
    )
    print(params, params2)

    fitted_function = lambda nparams: other_func(nparams, *params2)
    np.savetxt("fitted_function.txt", params2)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(parameter_count_list, highest_LR, label="Data")
    x = np.geomspace(5e4, 3e7, 100)
    plt.plot(x, power_law(x, *params), label="Power Law")
    plt.plot(x, fitted_function(x), label="Other function")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of parameters")
    plt.ylabel("Best Learning rate")
    plt.legend()
    plt.savefig("plots/fit_max_LR.pdf", bbox_inches="tight")


if __name__ == "__main__":
    fit_max_LR_func()
