import os
import torch
from tqdm import tqdm
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

import Transformer
from data_handling import normalize_data


def load_and_forecast(json_name, NN_path):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available() else "cpu"
        else "cpu"
    )
    print(f"Using {device}")

    with open(json_name, "r") as file:
        config = json.load(file)

    # Transformer parameters
    output_dim = config["transformer"]["output_dim"]
    d_model = config["transformer"]["d_model"]
    num_heads = config["transformer"]["num_heads"]
    num_layers = config["transformer"]["num_layers"]
    d_ff = config["transformer"]["d_ff"]
    dropout = config["transformer"]["dropout"]
    num_distribution_layers = config["transformer"]["num_distribution_layers"]
    max_seq_length = config["train"]["max_seq_length"]

    transformer = Transformer.Decoder_Transformer(
        output_dim,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        num_distribution_layers,
        device=device,
    ).to(device)
    num_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Number of parameters: ", num_params)
    print("Aspect ratio: ", d_model / num_layers)

    ####################################

    data_path = "../../TIME_opensource/final"
    path = data_path + "/weather/NOAA/NOAA_weather.npz"
    weather_data = np.load(path)

    # unnormalized_data = []
    data_list = []
    masks = []

    # for data_name in tqdm(weather_data.files):
    #     data = weather_data[data_name]

    #     for i in range(10):
    #         current_ts = data[i]
    #         new_data_length = current_ts.shape[0]
    #         mask = np.ones(new_data_length)

    #         # Need to append test and train masks
    #         data_list.append(
    #             normalize_data(current_ts, current_ts.mean(), current_ts.std())
    #         )
    #         masks.append(mask)

    ####################################
    import glob

    energy_paths = data_path + "/energy/"

    buildingbench_files = glob.glob(energy_paths + "*.npy")
    rnd_indices = np.random.choice(len(buildingbench_files), 2, replace=False)

    for rnd in tqdm(rnd_indices):
        buildingbench_data = np.load(buildingbench_files[rnd])

        for i in range(buildingbench_data.shape[0]):
            current_ts = buildingbench_data[i]
            new_data_length = current_ts.shape[0]
            mask = np.ones(new_data_length)

            # Need to append test and train masks
            data_list.append(
                normalize_data(current_ts, current_ts.mean(), current_ts.std())
            )
            # unnormalized_data.append(current_ts)
            masks.append(mask)

    ######################################

    # plt.plot(data_list[0])
    # plt.plot(unnormalized_data[0])
    # plt.show()
    # quit()

    index = 0
    y_train = (
        torch.tensor(data_list[index][:max_seq_length], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(device)
    )
    y_true = (
        torch.tensor(data_list[index][1 : max_seq_length + 1]).unsqueeze(-1).to(device)
    )

    output = transformer(y_train)
    plt.plot(y_true[:].detach().cpu(), zorder=20, color="k", label="True")
    mean = output[index, :, 0].detach().cpu()
    std = torch.sqrt(torch.nn.functional.softplus(output[index, :, 1].detach().cpu()))
    plt.plot(mean, color="r", label="Before training")
    plt.fill_between(
        np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.5, color="r"
    )

    # transformer.load_state_dict(torch.load(NN_path, map_location=torch.device("cpu")))
    # transformer.eval()

    output = transformer(y_train)
    mean = output[index, :, 0].detach().cpu()
    std = torch.sqrt(torch.nn.functional.softplus(output[index, :, 1].detach().cpu()))
    plt.plot(mean, color="b", label="After training")
    plt.fill_between(
        np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.5, color="b"
    )

    plt.legend()
    plt.savefig(f"plots/insequence_forecast_{num_params}.pdf", bbox_inches="tight")
    # plt.show()

    n_sequence = 109
    # data_to_forecast = (
    #     torch.tensor([data_list[index][:max_seq_length]], dtype=torch.float32)
    #     .unsqueeze(-1)
    #     .to(device)
    # )
    data_list = np.array(data_list)
    batch_size = 256

    data_to_forecast = torch.tensor(data_list, dtype=torch.float32).to(device)
    data_to_forecast_cut = data_to_forecast[:batch_size, :max_seq_length].unsqueeze(-1)
    forecast_truth = data_to_forecast[:batch_size, 1 : max_seq_length + 1]
    print(data_to_forecast_cut.shape, forecast_truth.shape)
    output = transformer(data_to_forecast_cut)
    print("MSE", transformer.MSE(output, forecast_truth).item())
    quit()

    data_to_forecast = torch.cat([data_to_forecast for _ in range(256)], dim=0)
    forecast = transformer.generate(data_to_forecast, n_sequence)
    std = np.std(forecast[:, :].detach().cpu().numpy(), axis=0)[:, 0]
    median = np.median(forecast[:, :].detach().cpu().numpy(), axis=0)[:, 0]

    forecast_xdim = np.arange(256, 256 + n_sequence)

    plt.plot(data_to_forecast[0, :256].detach().cpu(), color="k", ls="-")
    for i in range(1, 256):
        plt.plot(
            forecast_xdim,
            forecast[i].detach().cpu(),
            color="k",
            ls="-",
            alpha=0.2,
        )
    # plt.plot(
    #     forecast_xdim,
    #     median,
    #     color="k",
    #     ls="-",
    #     alpha=0.3,
    # )
    # plt.fill_between(
    #     forecast_xdim,
    #     median - std,
    #     median + std,
    #     color="k",
    #     alpha=0.3,
    # )
    plt.savefig("plots/forecast.pdf", bbox_inches="tight")
    # plt.show()

    return None


if __name__ == "__main__":
    json_name = "results/parameterscaling_19857411_studentT_training.json"
    NN_path = "results/parameterscaling_19857411_studentT_best.pt"
    # json_name = "results/parameterscaling_24451_studentT_training.json"
    # NN_path = "results/parameterscaling_24451_studentT_best.pt"

    load_and_forecast(json_name, NN_path)
