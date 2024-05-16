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

    data_path = "/home/tedwar42/data_tedwar42/final"

    data_list = []
    masks = []

    path = data_path + "/energy/buildings_900k_file0.npy"
    buildingbench_data = np.load(path)

    for i in range(buildingbench_data.shape[0]):
        current_ts = buildingbench_data[i]
        new_data_length = current_ts.shape[0]
        mask = np.ones(new_data_length)

        # Need to append test and train masks
        data_list.append(normalize_data(current_ts, current_ts.mean(), current_ts.std()))
        masks.append(mask)

    n_sequence = 256
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

    return mean, std_insequence, forecast, median, std, data_to_forecast, forecast_xdim, num_params

def plot_forecasts(models_info):
    fig, axs = plt.subplots(len(models_info), 1, figsize=(10, 15), sharex=True)
    
    for idx, (json_name, NN_path) in enumerate(models_info):
        mean, std_insequence, forecast, median, std, data_to_forecast, forecast_xdim, num_params = load_and_forecast(json_name, NN_path)

        n_sequence = 256
        max_seq_length = 256
        total_length = max_seq_length + n_sequence
        batch_size = 128
        
        x_total = np.arange(0, total_length)
        x_seq = np.arange(0, max_seq_length)
        axs[idx].plot(
            x_total,
            data_to_forecast[0].detach().cpu(),
            zorder=20,
            color="k",
            label="True",
            alpha=0.8,
        )
        axs[idx].axvline(x=max_seq_length, color="k", ls="--")

        axs[idx].plot(x_seq, mean, color="C1", label="In-sequence")
        axs[idx].fill_between(
            x_seq, mean - std_insequence, mean + std_insequence, alpha=0.5, color="C1"
        )

        axs[idx].plot(
            forecast_xdim,
            median,
            color="C2",
            ls="-",
            alpha=0.3,
            label="Forecast",
        )
        axs[idx].fill_between(
            forecast_xdim,
            median - std,
            median + std,
            color="C2",
            alpha=0.3,
        )
        axs[idx].set_ylim(-2.3,2.3)
        axs[idx].text(0.02, 0.95, f"$N_p$ =  {num_params}", transform=axs[idx].transAxes, verticalalignment='top')
        
    axs[0].legend(frameon = True, loc="lower left", ncol=3)
    axs[0].set_title("Building Bench Data")
    axs[3].set_xlabel("Token")
    fig.text(0.075, 0.5, 'Normalized Amplitude', ha='center', va='center', rotation='vertical', fontsize=20)
    plt.subplots_adjust(hspace=0.0)
    
    plt.savefig(f"plots/forecast_scaling.pdf", bbox_inches="tight")

if __name__ == "__main__":
    # [30787, 155011, 3825155, 5408259, 21433347]
    models_info = [
        ("results/parameterscaling_30787_studentT_training.json", "results/parameterscaling_30787_studentT_best.pt"),
        ("results/parameterscaling_155011_studentT_training.json", "results/parameterscaling_155011_studentT_best.pt"),
        ("results/parameterscaling_3825155_studentT_training.json", "results/parameterscaling_3825155_studentT_best.pt"),
        ("results/parameterscaling_21433347_studentT_training.json", "results/parameterscaling_21433347_studentT_best.pt"),
    ]
    plot_forecasts(models_info)
