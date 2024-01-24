import os
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from tqdm import tqdm
import yaml
import argparse
import json

# This is just until temporary implementation
import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

from data_handling import TimeSeriesDataset, download_data
from utils import convert_df_to_numpy, GradualWarmupScheduler
import Transformer


def train(config):
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    # Accessing the configuration values
    train_split = config["train"]["train_split"]
    max_seq_length = config["train"]["max_seq_length"]
    batch_size = config["train"]["batch_size"]
    test_batch_size = config["train"]["test_batch_size"]
    save = config["train"]["save"]
    total_training_steps = config["train"]["total_training_steps"]
    early_stopping = config["train"]["early_stopping"]
    warmup_steps = config["train"]["warmup_steps"]

    # Transformer parameters
    output_dim = config["transformer"]["output_dim"]
    d_model = config["transformer"]["d_model"]
    num_heads = config["transformer"]["num_heads"]
    num_layers = config["transformer"]["num_layers"]
    d_ff = config["transformer"]["d_ff"]
    dropout = config["transformer"]["dropout"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device}")

    # First lets download the data and make a data loader
    print("Downloading data...")
    datasets_to_load = {
        # Finance
        "nn5_weekly_dataset": "10.5281/zenodo.4656125",
        "nn5_daily_dataset_without_missing_values": "10.5281/zenodo.4656117",
        "bitcoin_dataset_without_missing_values": "10.5281/zenodo.5122101",
        "cif_2016_dataset": "10.5281/zenodo.4656042",
        "fred_md_dataset": "10.5281/zenodo.4654833",
        "dominick_dataset": "10.5281/zenodo.4654802",
        # Health
        "covid_mobility_dataset_without_missing_values": "10.5281/zenodo.4663809",
        "kdd_cup_2018_dataset_without_missing_values": "10.5281/zenodo.4656756",
        "covid_deaths_dataset": "10.5281/zenodo.4656009",
        "us_births_dataset": "10.5281/zenodo.4656049",
        "hospital_dataset": "10.5281/zenodo.4656014",
        # General
        "m4_hourly_dataset": "10.5281/zenodo.4656589",
        "m4_daily_dataset": "10.5281/zenodo.4656548",
        "m4_weekly_dataset": "10.5281/zenodo.4656522",
        "m4_monthly_dataset": "10.5281/zenodo.4656480",
        "m4_quarterly_dataset": "10.5281/zenodo.4656410",
        "m4_yearly_dataset": "10.5281/zenodo.4656379",
        "electricity_weekly_dataset": "10.5281/zenodo.4656141",
        "electricity_hourly_dataset": "10.5281/zenodo.4656140",
        "australian_electricity_demand_dataset": "10.5281/zenodo.4659727",
        "tourism_yearly_dataset": "10.5281/zenodo.4656103",
        "tourism_monthly_dataset": "10.5281/zenodo.4656096",
        "tourism_quarterly_dataset": "10.5281/zenodo.4656093",
        "elecdemand_dataset": "10.5281/zenodo.4656069",
        "car_parts_dataset_without_missing_values": "10.5281/zenodo.4656021",
        # Weather
        "oikolab_weather_dataset": "10.5281/zenodo.5184708",
        "sunspot_dataset_without_missing_values": "10.5281/zenodo.4654722",
        "solar_4_seconds_dataset": "10.5281/zenodo.4656027",
        "wind_4_seconds_dataset": "10.5281/zenodo.4656032",
        "weather_dataset": "10.5281/zenodo.4654822",
        "temperature_rain_dataset_without_missing_values": "10.5281/zenodo.5129091",
        "solar_weekly_dataset": "10.5281/zenodo.4656151",
        "solar_10_minutes_dataset": "10.5281/zenodo.4656144",
        "saugeenday_dataset": "10.5281/zenodo.4656058",
        "wind_farms_minutely_dataset_without_missing_values": "10.5281/zenodo.4654858",
        # Traffic
        "kaggle_web_traffic_weekly_dataset": "10.5281/zenodo.4656664",
        "kaggle_web_traffic_dataset_without_missing_values": "10.5281/zenodo.4656075",
        "pedestrian_counts_dataset": "10.5281/zenodo.4656626",
        "traffic_weekly_dataset": "10.5281/zenodo.4656135",
        "traffic_hourly_dataset": "10.5281/zenodo.4656132",
        "rideshare_dataset_without_missing_values": "10.5281/zenodo.5122232",
        "vehicle_trips_dataset_without_missing_values": "10.5281/zenodo.5122537",
        # Web
        "kaggle_web_traffic_dataset_without_missing_values": "10.5281/zenodo.4656075",
        "london_smart_meters_dataset_with_missing_values": "10.5281/zenodo.4656072",
    }
    dfs = download_data(datasets_to_load)

    (
        training_data_list,
        test_data_list,
        train_masks,
        test_masks,
    ) = convert_df_to_numpy(dfs, train_split)

    # num_cpus = os.cpu_count()

    train_dataset = TimeSeriesDataset(training_data_list, max_seq_length, train_masks)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_dataset = TimeSeriesDataset(test_data_list, max_seq_length, test_masks)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=4
    )

    print("Training dataset size: ", train_dataset.__len__())
    print("Test dataset size: ", test_dataset.__len__())

    print("Total number of training tokens:", train_dataset.total_length())

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
    learning_rate = 1e-3
    optimizer = optim.AdamW(transformer.parameters(), lr=learning_rate)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_training_steps, eta_min=1e-6
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        total_warmup_steps=warmup_steps,
        after_scheduler=cosine_scheduler,
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
    evaluation_interval = 100

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

            # Update tqdm bar with each step
            pbar.set_description(f"Step {step_counter}: Loss {loss.item():.5f}")
            pbar.update(1)

            loss.backward()
            optimizer.step()
            # print(f"Learning Rate = {scheduler.get_lr()[0]}")
            scheduler.step()

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
                    torch.save(
                        transformer.state_dict(),
                        f"transformer-{step_counter}.pt",
                    )

            step_counter += 1

        if patience_counter > early_stopping:
            print("Early stopping")
            break

    pbar.close()

    # Finally, lets save the losses
    file_name = f"results/transformer_{num_params}_training.json"
    model_file_name = f"results/transformer_{num_params}_model.pt"
    model_info = {
        "num_params": num_params,
        "train_losses": train_losses,
        "train_epochs": train_steps,
        "test_losses": test_losses,
        "test_epochs": test_steps,
        "datasets": list(datasets_to_load.keys()),
        "model_file_name": f"{model_file_name}",
    }
    # TODO: Eventually we want to save the best model
    print(f"Saving final model weights to {model_file_name}")
    torch.save(
        transformer.state_dict(),
        f"{model_file_name}.pt",
    )
    # Writing data to a JSON file
    with open(file_name, "w") as file:
        json.dump(model_info, file, indent=4)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script configuration.")
    parser.add_argument(
        "config_file", type=str, help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    train(args.config_file)
