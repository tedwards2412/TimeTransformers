import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib as mpl
import glob
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm

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


def aspect_ratio_interpolation():
    json_files = glob.glob("results/aspectratio*.json")
    print(json_files)
    N_list = []
    AR_list = []
    CRPS_list = []

    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)

        N_list.append(data["Nparams"])
        AR_list.append(
            data["transformer"]["d_model"] / data["transformer"]["num_layers"]
        )
        CRPS_list.append(min(data["CRPS_test_losses"]))
        print(
            "train losses",
            min(abs(np.diff(data["train_losses"]))),
            max(abs(np.diff(data["train_losses"]))),
        )
        print(
            "test losses",
            min(abs(np.diff(data["test_losses"]))),
            max(abs(np.diff(data["test_losses"]))),
        )

    # Convert lists to numpy arrays for easier handling
    N_array = np.array(N_list)
    AR_array = np.array(AR_list)
    CRPS_array = np.array(CRPS_list)

    # Create a regular grid where you want the interpolation
    AR_min, AR_max = 0.8, 1500
    CRPS_min, CRPS_max = 0.08, 0.18
    AR_grid, CRPS_grid = np.meshgrid(
        np.linspace(AR_min, AR_max, 500),
        np.linspace(CRPS_min, CRPS_max, 500),
    )

    # Perform the interpolation over the grid
    N_interpolated = griddata(
        points=(AR_array, CRPS_array),
        values=N_array,
        xi=(AR_grid, CRPS_grid),
        method="nearest",  # You can also try other methods like 'linear' or 'nearest'
    )

    # Plotting the interpolated data
    plt.figure(figsize=(7, 5))
    cm = mpl.colormaps["RdYlBu_r"]
    plt.contourf(
        AR_grid,
        CRPS_grid,
        N_interpolated,
        levels=2,
        cmap=cm,
        norm=LogNorm(vmin=10**4, vmax=10**8),
    )
    plt.colorbar(label="Number of Parameters")
    plt.xlabel("Aspect Ratio ($\mathrm{d}_m/ \mathrm{N}_l$)")
    plt.ylabel("Minimum CRPS")
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlim(AR_min, AR_max)
    plt.ylim(CRPS_min, CRPS_max)
    plt.savefig("plots/aspectratio_interpolated.pdf", bbox_inches="tight")


def aspect_ratio_plot():
    json_files = glob.glob("results/aspectratio*.json")
    print(json_files)
    N_list = []
    AR_list = []
    CRPS_list = []

    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)

        N_list.append(data["Nparams"])
        AR_list.append(
            data["transformer"]["d_model"] / data["transformer"]["num_layers"]
        )
        CRPS_list.append(min(data["CRPS_test_losses"]))

    plt.figure(figsize=(7, 5))
    cm = mpl.colormaps["RdYlBu_r"]
    sc = plt.scatter(
        AR_list,
        CRPS_list,
        c=N_list,
        s=35,
        cmap=cm,
        norm=LogNorm(vmin=10**4, vmax=10**8),
    )
    plt.colorbar(sc, label="Number of Parameters")
    plt.xlabel("Aspect Ratio ($\mathrm{d}_m/ \mathrm{N}_l$)")
    plt.ylabel("Minimum CRPS")
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlim(0.8, 1500)
    plt.ylim(0.08, 0.18)
    # plt.show()
    plt.savefig("plots/aspectratio_sensitivity.pdf", bbox_inches="tight")


def aspect_ratio_plot_inverted():
    json_files = glob.glob("results/aspectratio*.json")
    # 155011, 3825155, 21433347
    # json_files = [
    #     "results/parameterscaling_155011_studentT_training.json",
    #     "results/parameterscaling_3825155_studentT_training.json",
    #     "results/parameterscaling_21433347_studentT_training.json",
    # ]
    print(json_files)
    N_list = []
    AR_list = []
    CRPS_list = []

    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)

        N_list.append(data["Nparams"])
        AR_list.append(
            data["transformer"]["d_model"] / data["transformer"]["num_layers"]
        )
        CRPS_list.append(min(data["CRPS_test_losses"]))

    plt.figure(figsize=(7, 5))
    cm = mpl.colormaps["RdYlBu_r"]
    sc = plt.scatter(
        N_list,
        CRPS_list,
        c=AR_list,
        s=35,
        cmap=cm,
        norm=LogNorm(vmin=0.8, vmax=1500),
    )
    plt.colorbar(sc, label="Aspect Ratio ($\mathrm{d}_m/ \mathrm{N}_l$)")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Minimum CRPS")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e3, 1e7)
    plt.ylim(0.1, 0.4)
    # plt.show()
    plt.savefig("plots/aspectratio_sensitivity_inverted.pdf", bbox_inches="tight")


if __name__ == "__main__":
    aspect_ratio_interpolation()
    aspect_ratio_plot()
    # aspect_ratio_plot_inverted()
