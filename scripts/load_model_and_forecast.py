import os
import torch
from tqdm import tqdm
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

import Transformer
from data_handling import normalize_data, TimeSeriesDataset, load_datasets


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

    model_state = torch.load(NN_path, map_location=device)
    transformer.load_state_dict(model_state, strict=True)
    # transformer.load_state_dict(torch.load(NN_path, map_location=torch.device("cpu")))
    transformer.to(device)
    transformer.eval()

    ######################################
    # test = 0
    # end_test = 5
    # total_MSE_test_loss = 0
    # total_test_samples = 0
    # for batch in train_dataloader:
    #     train, true, mask = batch
    #     current_batch_size = train.shape[0]
    #     batched_data = train.to(device)
    #     batched_data_true = true.to(device)
    #     output = transformer(batched_data)
    #     test_loss_MSE = transformer.MSE(output, batched_data_true, mask=mask.to(device))
    #     total_MSE_test_loss += test_loss_MSE.item() * current_batch_size
    #     test += 1
    #     total_test_samples += current_batch_size
    #     if test == end_test:
    #         break

    # average_MSE_test_loss = total_MSE_test_loss / total_test_samples
    # # plt.plot(train[0, :, 0].detach().cpu())
    # # plt.plot(true[0, :].detach().cpu())
    # # plt.show()
    # # quit()
    # print("Test loss: ", average_MSE_test_loss)
    # quit()

    # batch_size = config["train"]["batch_size"]
    # datasets_to_load = config["datasets"]
    # train_split = config["train"]["train_split"]
    # datasets_to_load["monash"] = ["electricity_hourly_dataset"]
    # datasets_to_load["finance"] = []
    # datasets_to_load["energy"] = []
    # datasets_to_load["science"] = []
    # datasets_to_load["audio"] = []
    # datasets_to_load["traffic"] = []
    # datasets_to_load["traffic"] = ["NOAA_dataset"]

    # print("Loading data...")
    # (
    #     training_data_list,
    #     test_data_list,
    #     train_masks,
    #     test_masks,
    # ) = load_datasets(datasets_to_load, train_split)

    # train_dataset = TimeSeriesDataset(training_data_list, max_seq_length, train_masks)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # print("Training dataset size: ", train_dataset.__len__())
    # print("Total number of training tokens:", train_dataset.total_length())

    # for batch in train_dataloader:
    #     train, true, mask = batch
    #     break

    # batched_data = train.to(device)
    # batched_data_true = true.to(device)
    # output = transformer(batched_data)
    # print("evaluating model")

    # index = 0
    # plt.plot(
    #     batched_data_true[index, :].detach().cpu(), zorder=20, color="k", label="True"
    # )
    # mean = output[index, :, 0].detach().cpu()
    # std = torch.sqrt(torch.nn.functional.softplus(output[index, :, 1].detach().cpu()))
    # plt.plot(mean, color="r", label="Best model")
    # plt.fill_between(
    #     np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.5, color="r"
    # )
    # plt.legend()
    # plt.savefig(f"plots/insequence_forecast_{num_params}.pdf", bbox_inches="tight")
    # quit()
    # plt.show()

    ################ Now lets forecast ################

    # ####################################
    # import glob

    # energy_paths = data_path + "/energy/"

    # buildingbench_files = glob.glob(energy_paths + "*.npy")
    # rnd_indices = np.random.choice(len(buildingbench_files), 2, replace=False)

    # for rnd in tqdm(rnd_indices):
    #     buildingbench_data = np.load(buildingbench_files[rnd])

    #     for i in range(buildingbench_data.shape[0]):
    #         current_ts = buildingbench_data[i]
    #         new_data_length = current_ts.shape[0]
    #         mask = np.ones(new_data_length)

    #         # Need to append test and train masks
    #         data_list.append(
    #             normalize_data(current_ts, current_ts.mean(), current_ts.std())
    #         )
    #         # unnormalized_data.append(current_ts)
    #         masks.append(mask)
    ####################################

    data_path = "../../TIME_opensource/final"
    path = data_path + "/weather/NOAA/NOAA_weather.npz"
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

    n_sequence = 109
    index = 0
    data_arr = np.array(data_list[index])
    data_to_forecast = (
        torch.tensor([data_arr[:max_seq_length]], dtype=torch.float32)
        .unsqueeze(-1)
        .to(device)
    )
    data_to_forecast = torch.cat([data_to_forecast for _ in range(256)], dim=0)
    print(data_arr.shape, data_to_forecast.shape)

    forecast = transformer.generate(data_to_forecast, n_sequence)
    std = np.std(forecast[:, :].detach().cpu().numpy(), axis=0)[:, 0]
    median = np.median(forecast[:, :].detach().cpu().numpy(), axis=0)[:, 0]
    forecast_xdim = np.arange(256, 256 + n_sequence)

    plt.plot(data_arr, color="k", ls="-")
    for i in range(1, 256):
        plt.plot(
            forecast_xdim,
            forecast[i].detach().cpu(),
            color="k",
            ls="-",
            alpha=0.2,
        )
    plt.plot(
        forecast_xdim,
        median,
        color="k",
        ls="-",
        alpha=0.3,
    )
    plt.fill_between(
        forecast_xdim,
        median - std,
        median + std,
        color="k",
        alpha=0.3,
    )
    plt.savefig("plots/forecast.pdf", bbox_inches="tight")
    # plt.show()

    return None


if __name__ == "__main__":
    json_name = "results/parameterscaling_19857411_studentT_training.json"
    NN_path = "results/parameterscaling_21433347_studentT_best.pt"
    # parameterscaling_21433347_studentT_best.pt
    # json_name = "results/parameterscaling_24451_studentT_training.json"
    # NN_path = "results/parameterscaling_24451_studentT_best.pt"

    load_and_forecast(json_name, NN_path)
