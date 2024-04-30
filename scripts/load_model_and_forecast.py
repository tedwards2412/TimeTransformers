import os
import torch
from tqdm import tqdm
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

plt.style.use("plots.mplstyle")

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

import Transformer
from data_handling import normalize_data, TimeSeriesDataset, load_datasets


def load_and_forecast(json_name, NN_path):

    ####### LOAD MODEL #######
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model_state = torch.load(NN_path, map_location=device)
    transformer.load_state_dict(model_state, strict=True)
    transformer.to(device)
    transformer.eval()

    ####### LOAD DATA #######

    data_path = "/Users/thomasedwards/Dropbox/Work/ML/TIME_opensource/final"
    path = data_path + "/weather/ERA5/ERA5.npz"
    weather_data = np.load(path)

    data_list = []
    masks = []

    for data_name in tqdm(weather_data.files):
        data = weather_data[data_name]

        for i in range(5):
            current_ts = data[i]
            new_data_length = current_ts.shape[0]
            mask = np.ones(new_data_length)

            # Need to append test and train masks
            data_list.append(
                normalize_data(current_ts, current_ts.mean(), current_ts.std())
            )
            masks.append(mask)

    n_sequence = 10
    total_length = max_seq_length + n_sequence
    index = 5
    batch_size = 128

    data_arr = torch.tensor(data_list[index]).to(device)
    data_to_forecast = torch.tensor(
        data_arr[:total_length].unsqueeze(0), dtype=torch.float32
    ).to(device)
    batched_forecast_data = torch.cat(
        [data_to_forecast[:, :max_seq_length] for _ in range(batch_size)], dim=0
    ).unsqueeze(-1)

    ####### FORECAST #######

    output = transformer(batched_forecast_data)

    mean = output[0, :, 0].detach().cpu()
    std_insequence = torch.sqrt(
        torch.nn.functional.softplus(output[0, :, 1].detach().cpu())
    )

    forecast = transformer.generate(batched_forecast_data, n_sequence)
    std = np.std(forecast[:, :].detach().cpu().numpy(), axis=0)[:, 0]
    median = np.median(forecast[:, :].detach().cpu().numpy(), axis=0)[:, 0]
    forecast_xdim = np.arange(max_seq_length, max_seq_length + n_sequence)

    ####### PLOT EVERYTHING #######

    plt.figure(figsize=(10, 5))
    x_total = np.arange(0, total_length)
    x_seq = np.arange(0, max_seq_length)
    plt.plot(
        x_total,
        data_to_forecast[0].detach().cpu(),
        zorder=20,
        color="k",
        label="True",
        alpha=0.2,
    )
    plt.axvline(x=max_seq_length, color="k", ls="--", label="Forecast")

    plt.plot(x_seq, mean, color="C1", label="Best model")
    plt.fill_between(
        x_seq, mean - std_insequence, mean + std_insequence, alpha=0.5, color="C1"
    )

    for i in range(batch_size):
        plt.plot(
            forecast_xdim,
            forecast[i].detach().cpu(),
            color="C2",
            ls="-",
            alpha=0.01,
        )
    plt.plot(
        forecast_xdim,
        median,
        color="C2",
        ls="-",
        alpha=0.3,
        label="Forecast",
    )
    plt.fill_between(
        forecast_xdim,
        median - std,
        median + std,
        color="k",
        alpha=0.3,
    )
    # plt.savefig(f"plots/forecast_{num_params}.pdf", bbox_inches="tight")
    plt.legend()
    plt.show()

    return None


if __name__ == "__main__":
    json_name = "results/maxLRscaling_18277379_0.00070711.json"
    NN_path = "results/maxLRscaling_18277379_0.00070711_best.pt"
    load_and_forecast(json_name, NN_path)
