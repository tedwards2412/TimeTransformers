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


def max_LR_plot():
    json_files = glob.glob("results/maxLRscaling*.json")
    print(len(json_files))
    N_list = []
    maxLR_list = []
    CRPS_list = []
    divege_list = []

    for json_file in json_files:
        # print(json_file)
        with open(json_file, "r") as file:
            data = json.load(file)

        try:
            divege_list.append(data["diverge"])
        except:
            continue
        N_list.append(data["Nparams"])
        maxLR_list.append(data["train"]["max_LR"])
        CRPS_list.append(min(data["CRPS_test_losses"]))

    N_list = np.array(N_list)
    maxLR_list = np.array(maxLR_list)
    divege_list = np.array(divege_list)
    CRPS_list = np.array(CRPS_list)

    plt.figure(figsize=(7, 5))
    cm = mpl.colormaps["RdYlBu_r"]
    sc = plt.scatter(
        maxLR_list[~divege_list],
        CRPS_list[~divege_list],
        c=N_list[~divege_list],
        s=35,
        cmap=cm,
        norm=LogNorm(vmin=10**5, vmax=10**7),
    )
    # plt.colorbar(sc, label="Number of Parameters")

    sc = plt.scatter(
        maxLR_list[divege_list],
        CRPS_list[divege_list],
        c=N_list[divege_list],
        s=35,
        marker="x",
        cmap=cm,
        norm=LogNorm(vmin=10**5, vmax=10**7),
    )
    plt.colorbar(sc, label="Number of Parameters")

    plt.xlabel("Maximum Learning Rate")
    plt.ylabel("Minimum CRPS")
    plt.xscale("log")
    # plt.yscale("log")
    # plt.xlim(0.8, 1500)
    plt.ylim(0.06, 0.16)
    # plt.show()
    plt.savefig("plots/maxLR.pdf", bbox_inches="tight")


if __name__ == "__main__":
    max_LR_plot()
