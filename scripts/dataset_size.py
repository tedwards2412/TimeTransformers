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


def train():
    train_split = 0.8
    max_seq_length = 1024
    batch_size = 512
    test_batch_size = 1024

    # First lets download the data and make a data loader
    print("Downloading data...")
    datasets_to_load = {
        # Finance
        "nn5_weekly_dataset": "10.5281/zenodo.4656125",
        "nn5_daily_dataset_without_missing_values": "10.5281/zenodo.4656117",
        "bitcoin_dataset_without_missing_values": "10.5281/zenodo.5122101",
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
        # Weather
        "oikolab_weather_dataset": "10.5281/zenodo.5184708",
        "sunspot_dataset_without_missing_values": "10.5281/zenodo.4654722",
        "solar_4_seconds_dataset": "10.5281/zenodo.4656027",
        "wind_4_seconds_dataset": "10.5281/zenodo.4656032",
        "weather_dataset": "10.5281/zenodo.4654822",
        "temperature_rain_dataset_without_missing_values": "10.5281/zenodo.5129091",
        # Traffic
        "kaggle_web_traffic_weekly_dataset": "10.5281/zenodo.4656664",
        "kaggle_web_traffic_dataset_without_missing_values": "10.5281/zenodo.4656075",
        "pedestrian_counts_dataset": "10.5281/zenodo.4656626",
        "traffic_weekly_dataset": "10.5281/zenodo.4656135",
        "traffic_hourly_dataset": "10.5281/zenodo.4656132",
        "rideshare_dataset_without_missing_values": "10.5281/zenodo.5122232",
        "vehicle_trips_dataset_without_missing_values": "10.5281/zenodo.5122537",
        # Web
        "web_traffic_extended_dataset_without_missing_values": "10.5281/zenodo.7371038",
        "london_smart_meters_dataset_with_missing_values": "10.5281/zenodo.4656072",
    }
    # print(len(datasets_to_load.keys()))
    # quit()
    dfs = download_data(datasets_to_load)

    training_data_list, test_data_list = convert_df_to_numpy(dfs, train_split)

    train_dataset = TimeSeriesDataset(training_data_list, max_seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TimeSeriesDataset(test_data_list, max_seq_length)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    print("Training dataset size: ", train_dataset.__len__())
    print("Test dataset size: ", test_dataset.__len__())

    print("Total number of training tokens:", train_dataset.total_length())
    print("Total number of test tokens:", train_dataset.total_length())

    return None


if __name__ == "__main__":
    train()
