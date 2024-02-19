import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import random

# This is just until temporary implementation
import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

from utils import convert_tsf_to_dataframe


def normalize_data(data, mean, std):
    if std == 0:
        return data - mean
    else:
        return (data - mean) / std


class TimeSeriesDataset(Dataset):
    def __init__(
        self, data, max_sequence_length, miss_vals_mask, test=False, test_size=0.2
    ):
        self.max_sequence_length = max_sequence_length
        self.means = np.array([np.mean(data[i]) for i in range(len(data))])
        self.std = np.array([np.std(data[i]) for i in range(len(data))])
        self.miss_vals_mask = miss_vals_mask

        self.data = [
            normalize_data(data[i], self.means[i], self.std[i])
            for i in range(len(data))
        ]
        self.probs = (
            np.array([len(self.data[i]) for i in range(len(self.data))])
            / self.total_length()
        )
        self.test = test
        self.test_size = test_size
        self.data_len = self.calc_len(self.test, self.test_size)

    def __len__(self):
        return self.data_len

    def calc_len(self, test, test_size):
        l = 0
        for i in range(len(self.data)):
            l += len(self.data[i])
        length = int(l / self.max_sequence_length)
        if test:
            return int(length * test_size)
        else:
            return length

    def total_length(self):
        l = 0
        for i in range(len(self.data)):
            l += len(self.data[i])
        return l

    def __getitem__(self, idx):
        # I will just randomly select one of the time series
        # and then randomly select a subsequence of length max_sequence_length
        new_idx = np.random.choice(a=len(self.probs), p=self.probs)
        series = self.data[new_idx]
        series_mask = self.miss_vals_mask[new_idx]
        if len(series) > self.max_sequence_length:
            # Randomly select a starting point for the sequence
            start_index = random.randint(0, len(series) - self.max_sequence_length - 1)

            # Slice the series to get a random subsequence of length max_sequence_length
            train_series = torch.tensor(
                series[start_index : start_index + self.max_sequence_length],
                dtype=torch.float32,
            ).unsqueeze(-1)
            true_series = torch.tensor(
                series[start_index + 1 : start_index + self.max_sequence_length + 1],
                dtype=torch.float32,
            )
            missing_vals_batch = torch.tensor(
                series_mask[start_index : start_index + self.max_sequence_length],
                dtype=torch.bool,
            )
            mask = torch.ones_like(train_series, dtype=torch.bool).squeeze(-1)
            final_mask = mask & missing_vals_batch

            return (
                train_series,
                true_series,
                final_mask,
            )

        else:
            train_series = torch.tensor(
                series[:-1],
                dtype=torch.float32,
            )
            true_series = torch.tensor(series[1:], dtype=torch.float32)

            # mask = torch.ones(len(train_series), dtype=torch.bool)
            mask = torch.tensor(series_mask[:-1], dtype=torch.bool)

            # Calculate the number of padding elements needed
            padding_length = self.max_sequence_length - len(train_series)

            # Create padding tensors
            series_padding = torch.zeros(padding_length)
            mask_padding = torch.zeros(padding_length, dtype=torch.bool)

            # Concatenate the original tensors with their respective paddings
            train_series = torch.cat([train_series, series_padding])
            true_series = torch.cat([true_series, series_padding])
            mask = torch.cat([mask, mask_padding])

            return (
                train_series.unsqueeze(-1),
                true_series,
                mask,
            )


def download_single_datafile(dataset_name, dataset_id):
    # Define the path for the zip file
    tsf_file_path = f"../data/{dataset_name}.tsf"
    zip_file_path = f"../data/{dataset_name}.zip"

    # Check if the dataset already exists
    if not os.path.exists(tsf_file_path):
        # Download the dataset if it doesn't exist
        os.system(f"zenodo_get {dataset_id}")
        os.system(f"mv {dataset_name}.zip {zip_file_path}")
        print(f"Downloaded {dataset_name}.zip")

        # Unzip the dataset
        os.system(f"unzip -o {zip_file_path} -d ../data/")
        print(f"Unzipped {dataset_name}.zip")

        # Remove the zip file
        os.system(f"rm {zip_file_path}")
        os.system(f"rm md5sums.txt")
    else:
        print(f"{dataset_name}.tsf already exists. Skipping download.")

    # Convert the tsf file to a pandas dataframe
    return convert_tsf_to_dataframe(f"../data/{dataset_name}.tsf")[0]


def download_data(dataset_names):
    df_list = []
    for dataset_name in dataset_names:
        dataset_id = dataset_dict[dataset_name]
        df_list.append(download_single_datafile(dataset_name, dataset_id))

    return df_list


dataset_dict = {
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
    "pedestrian_counts_dataset": "10.5281/zenodo.4656626",
    "traffic_weekly_dataset": "10.5281/zenodo.4656135",
    "traffic_hourly_dataset": "10.5281/zenodo.4656132",
    "rideshare_dataset_without_missing_values": "10.5281/zenodo.5122232",
    "vehicle_trips_dataset_without_missing_values": "10.5281/zenodo.5122537",
    # Web
    "kaggle_web_traffic_weekly_dataset": "10.5281/zenodo.4656664",
    "kaggle_web_traffic_dataset_without_missing_values": "10.5281/zenodo.4656075",
    "london_smart_meters_dataset_with_missing_values": "10.5281/zenodo.4656072",
}
