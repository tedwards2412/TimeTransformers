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
plt.style.use("plots.mplstyle")


def nheads_plot():
    json_files = glob.glob("results/nheads*.json")
    print(len(json_files))
    N_list = []
    nheads = []
    CRPS_list = []

    for json_file in json_files:
        # print(json_file)
        with open(json_file, "r") as file:
            data = json.load(file)

        N_list.append(data["Nparams"])
        nheads.append(data["transformer"]["num_heads"])
        CRPS_list.append(min(data["CRPS_test_losses"]))

    N_list = np.array(N_list)
    nheads = np.array(nheads)
    CRPS_list = np.array(CRPS_list)

    plt.figure(figsize=(7, 5))
    cm = mpl.colormaps["RdYlBu_r"]

    # plt.colorbar(sc, label="Number of Parameters")

    sc = plt.scatter(
        nheads,
        CRPS_list,
        c=N_list,
        s=35,
        cmap=cm,
        norm=LogNorm(vmin=10**5, vmax=10**8),
    )
    plt.colorbar(sc, label="Number of Parameters")

    plt.xlabel("Number of Heads")
    plt.ylabel("Minimum CRPS")
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlim(1, 50)
    plt.ylim(0.06, 0.16)
    # plt.show()
    plt.savefig("plots/n_heads.pdf", bbox_inches="tight")


def nheads_aspectratio_combined():
    # First plot
    json_files = glob.glob("results/nheads*.json")
    print(len(json_files))
    N_list = []
    nheads = []
    CRPS_list = []

    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)

        N_list.append(data["Nparams"])
        nheads.append(data["transformer"]["num_heads"])
        CRPS_list.append(min(data["CRPS_test_losses"]))

    N_list = np.array(N_list)
    print(N_list.max() > 1e8)
    print(N_list.min() < 1e4)
    nheads = np.array(nheads)
    CRPS_list = np.array(CRPS_list)

    # Second plot
    json_files = glob.glob("results/aspectratio*.json")
    print(json_files)
    N_list2 = []
    AR_list = []
    CRPS_list2 = []
    msize = 100

    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)

        N_list2.append(data["Nparams"])
        AR_list.append(
            data["transformer"]["d_model"] / data["transformer"]["num_layers"]
        )
        CRPS_list2.append(min(data["CRPS_test_losses"]))

    # Convert lists to numpy arrays for easier handling
    N_list2 = np.array(N_list2)
    print(N_list2.max() > 1e8)
    print(N_list2.min() < 1e4)
    AR_list = np.array(AR_list)
    CRPS_list2 = np.array(CRPS_list2)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    # First subplot
    cm = mpl.colormaps["inferno_r"]
    sc1 = ax2.scatter(
        nheads,
        CRPS_list,
        c=N_list,
        s=msize,
        cmap=cm,
        norm=LogNorm(vmin=10**4, vmax=10**8),
    )
    ax2.set_xlabel("Number of Heads, $N_{\mathrm{heads}}$")
    ax2.set_xscale("log")
    ax2.set_xlim(0.8, 50)
    ax2.set_ylim(0.08, 0.18)

    # Second subplot
    sc2 = ax1.scatter(
        AR_list,
        CRPS_list2,
        c=N_list2,
        s=msize,
        cmap=cm,
        norm=LogNorm(vmin=10**4, vmax=10**8),
    )

    ax1.set_xlabel("Aspect Ratio ($d_\mathrm{m}/ N_\mathrm{l}$)")
    ax1.set_xscale("log")

    # Adding colorbar and labels
    cbar = fig.colorbar(sc1, ax=[ax1, ax2], label="Number of Parameters", pad=0.02)
    ax1.set_ylabel("Minimum CRPS")
    plt.savefig("plots/n_heads_AR_combined.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    nheads_plot()
    nheads_aspectratio_combined()
