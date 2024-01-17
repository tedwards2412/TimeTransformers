import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import torch
import random
import torch.optim as optim
from tqdm import tqdm
import json

# This is just until temporary implementation
import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

from data_handling import TimeSeriesDataset, download_data
from utils import convert_df_to_numpy
import Transformer


def train():
    train_split = 0.8
    max_seq_length = 1024
    batch_size = 512
    test_batch_size = 512
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device}")
    save = False
    total_training_steps = 2.5e5
    early_stopping = 2.5e5

    # Transformer parameters
    output_dim = 2  # To begin with we can use a Gaussian with mean and variance
    d_model = 16
    num_heads = 1
    num_layers = 2
    d_ff = 16
    dropout = 0.1

    # First lets download the data and make a data loader
    print("Downloading data...")
    datasets_to_load = {
        "oikolab_weather_dataset": "10.5281/zenodo.5184708",
        # "covid_deaths_dataset": "10.5281/zenodo.4656009",
        # "us_births_dataset": "10.5281/zenodo.4656049",
        # "solar_4_seconds_dataset": "10.5281/zenodo.4656027",
        # "wind_4_seconds_dataset": "10.5281/zenodo.4656032",
        # "weather_dataset": "10.5281/zenodo.4654822",
        "hospital_dataset": "10.5281/zenodo.4656014",
        # "electricity_hourly_dataset": "10.5281/zenodo.4656140",
        # "traffic_hourly_dataset": "10.5281/zenodo.4656132",
        # "rideshare_dataset_without_missing_values": "10.5281/zenodo.5122232",
        # "bitcoin_dataset_without_missing_values": "10.5281/zenodo.5122101",
        # "australian_electricity_demand_dataset": "10.5281/zenodo.4659727",
        # "sunspot_dataset_without_missing_values": "10.5281/zenodo.4654722",
        # "london_smart_meters_dataset_with_missing_values": "10.5281/zenodo.4656072",
    }
    dfs = download_data(datasets_to_load)

    training_data_list, test_data_list, train_masks, test_masks = convert_df_to_numpy(
        dfs, train_split
    )

    train_dataset = TimeSeriesDataset(training_data_list, max_seq_length, train_masks)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TimeSeriesDataset(test_data_list, max_seq_length, test_masks)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    print("Training dataset size: ", train_dataset.__len__())
    print("Test dataset size: ", test_dataset.__len__())

    print("Total number of training tokens:", train_dataset.total_length())
    # quit()

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
    num_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Number of parameters: ", num_params)

    # Now lets train it!
    learning_rate = 0.0001
    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=500, verbose=True, factor=0.5
    )
    transformer.train()

    train_steps = []
    train_losses = []

    test_steps = []
    test_losses = []

    transformer.train()
    min_loss = 1e10
    patience_counter = 0

    step_counter = 0
    evaluation_interval = 10

    # Initialize tqdm progress bar
    pbar = tqdm(total=total_training_steps, desc="Training", position=0)

    while step_counter < total_training_steps:
        transformer.train()
        for batch in train_dataloader:
            if step_counter >= total_training_steps:
                break

            train, true, mask = batch

            batched_data = train.to(device)
            batched_data_true = true.to(device)
            optimizer.zero_grad()
            output = transformer(batched_data, custom_mask=mask.to(device))
            loss = transformer.Gaussian_loss(output, batched_data_true)
            train_losses.append(loss.item())
            train_steps.append(step_counter)

            step_counter += 1

            # Update tqdm bar with each step
            pbar.set_description(f"Step {step_counter}: Loss {loss.item():.5f}")
            pbar.update(1)

            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        if step_counter % evaluation_interval == 0:
            transformer.eval()
            total_test_loss = 0
            with torch.no_grad():  # Disable gradient calculation
                for batch in test_dataloader:
                    train, true, mask = batch
                    batched_data = train.to(device)
                    batched_data_true = true.to(device)
                    output = transformer(batched_data)
                    test_loss = transformer.Gaussian_loss(output, batched_data_true)
                    total_test_loss += test_loss.item()

            average_test_loss = total_test_loss / len(test_dataloader)
            if average_test_loss < min_loss:
                min_loss = average_test_loss
                patience_counter = 0
            else:
                patience_counter += 1

            test_losses.append(average_test_loss)
            test_steps.append(step_counter)

            if save:
                torch.save(transformer.state_dict(), f"transformer-{step_counter}.pt")

        if patience_counter > early_stopping:
            print("Early stopping")
            break

    pbar.close()

    # Finally, lets save the losses
    file_name = f"results/transformer_{num_params}_training.json"
    model_info = {
        "num_params": num_params,
        "train_losses": train_losses,
        "train_epochs": train_steps,
        "test_losses": test_losses,
        "test_epochs": test_steps,
        "datasets": list(datasets_to_load.keys()),
    }
    # Writing data to a JSON file
    with open(file_name, "w") as file:
        json.dump(model_info, file, indent=4)

    return None


if __name__ == "__main__":
    train()
