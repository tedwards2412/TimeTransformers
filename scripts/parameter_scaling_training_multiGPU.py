import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import torch
import random
import torch.optim as optim
from tqdm import tqdm
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# This is just until temporary implementation
import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

from data_handling import TimeSeriesDataset, download_data
from utils import convert_df_to_numpy
import Transformer


def Gaussian_loss(
    transformer_pred, y_true, epsilon=torch.tensor(1e-6, dtype=torch.float32)
):
    epsilon = epsilon.to(transformer_pred.device)
    # Splitting the output into mean and variance
    mean = transformer_pred[:, :, 0]
    var = torch.nn.functional.softplus(transformer_pred[:, :, 1]) + epsilon

    # Calculating the Gaussian negative log-likelihood loss
    # print(y_true, mean, torch.log(var))
    loss = torch.mean((y_true - mean) ** 2 / var + torch.log(var))

    return loss


def train(local_rank):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    train_split = 0.8
    max_seq_length = 512
    batch_size = 512
    test_batch_size = 1024
    save = False
    epochs = 51
    learning_rate = 0.0001
    early_stopping = 10

    # Transformer parameters
    output_dim = 2  # To begin with we can use a Gaussian with mean and variance
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 128
    dropout = 0.0

    # First lets download the data and make a data loader
    print("Downloading data...")
    datasets_to_load = {
        # "oikolab_weather_dataset": "10.5281/zenodo.5184708",
        # "covid_deaths_dataset": "10.5281/zenodo.4656009",
        # "us_births_dataset": "10.5281/zenodo.4656049",
        # "solar_4_seconds_dataset": "10.5281/zenodo.4656027",
        # "wind_4_seconds_dataset": "10.5281/zenodo.4656032",
        # "weather_dataset": "10.5281/zenodo.4654822",
        # "hospital_dataset": "10.5281/zenodo.4656014",
        # "electricity_hourly_dataset": "10.5281/zenodo.4656140",
        # "traffic_hourly_dataset": "10.5281/zenodo.4656132",
        "rideshare_dataset_without_missing_values": "10.5281/zenodo.5122232",
        # "bitcoin_dataset_without_missing_values": "10.5281/zenodo.5122101",
        # "australian_electricity_demand_dataset": "10.5281/zenodo.4659727",
        # "sunspot_dataset_without_missing_values": "10.5281/zenodo.4654722",
        # "london_smart_meters_dataset_with_missing_values": "10.5281/zenodo.4656072",
    }
    dfs = download_data(datasets_to_load)

    training_data_list, test_data_list = convert_df_to_numpy(dfs, train_split)

    # train_dataset = TimeSeriesDataset(training_data_list, max_seq_length)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test_dataset = TimeSeriesDataset(test_data_list, max_seq_length)
    # test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    train_dataset = TimeSeriesDataset(training_data_list, max_seq_length)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler
    )

    test_dataset = TimeSeriesDataset(test_data_list, max_seq_length)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, sampler=test_sampler
    )

    print("Training dataset size: ", train_dataset.__len__())
    print("Test dataset size: ", test_dataset.__len__())

    # Now lets make a transformer

    transformer = Transformer.Decoder_Transformer(
        output_dim,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        device=device,
    ).to(device)
    transformer = DDP(transformer, device_ids=[local_rank])
    num_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Number of parameters: ", num_params)

    # Now lets train it!

    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
    transformer.train()

    e_counter = 1
    train_epochs = []
    train_losses = []

    test_epochs = []
    test_losses = []

    pbar = tqdm(range(epochs))
    transformer.train()
    min_loss = 1e10
    patience_counter = 0

    for epoch in pbar:
        train_sampler.set_epoch(epoch)
        transformer.train()
        for batch in train_dataloader:
            train, true, mask = batch
            batched_data = train.to(device)
            batched_data_true = true.to(device)
            optimizer.zero_grad()
            output = transformer(batched_data, custom_mask=mask.to(device))
            loss = Gaussian_loss(output, batched_data_true)
            train_losses.append(loss.item())
            train_epochs.append(e_counter)

            pbar.set_description(f"Epoch {epoch}: Loss {loss.item():.5f},")

            loss.backward()
            optimizer.step()
            e_counter += 1

        if epoch % 5 == 0:
            transformer.eval()
            total_test_loss = 0
            with torch.no_grad():  # Disable gradient calculation
                for batch in test_dataloader:
                    train, true, mask = batch
                    batched_data = train.to(device)
                    batched_data_true = true.to(device)
                    output = transformer(batched_data)
                    test_loss = Gaussian_loss(output, batched_data_true)
                    total_test_loss += test_loss.item()

            average_test_loss = total_test_loss / len(test_dataloader)
            if average_test_loss < min_loss:
                min_loss = average_test_loss
                patience_counter = 0
            else:
                patience_counter += 1

            test_losses.append(average_test_loss)
            test_epochs.append(e_counter)

            if save:
                torch.save(transformer.state_dict(), f"transformer-{epoch}.pt")

        if patience_counter > early_stopping:
            print("Early stopping")
            break

    # Finally, lets save the losses
    file_name = f"results/transformer_{num_params}_training.json"
    model_info = {
        "num_params": num_params,
        "train_losses": train_losses,
        "train_epochs": train_epochs,
        "test_losses": test_losses,
        "test_epochs": test_epochs,
        "datasets": list(datasets_to_load.keys()),
    }
    # Writing data to a JSON file
    with open(file_name, "w") as file:
        json.dump(model_info, file, indent=4)

    return None


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    train(local_rank)
    # train()
