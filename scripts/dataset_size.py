import os
from torch.utils.data import DataLoader

# This is just until temporary implementation
import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

from data_handling import TimeSeriesDataset, download_data
from utils import convert_df_to_numpy


def train():
    train_split = 0.95
    max_seq_length = 1024
    batch_size = 512
    test_batch_size = 4096

    # First lets download the data and make a data loader
    print("Downloading data...")

    datasets_to_load = [
        "electricity_hourly_dataset",
        "traffic_hourly_dataset",
        "traffic_weekly_dataset",
        "solar_4_seconds_dataset",
        "solar_weekly_dataset",
        "solar_10_minutes_dataset",
        "wind_4_seconds_dataset",
        "oikolab_weather_dataset",
        "nn5_weekly_dataset",
        "nn5_daily_dataset_without_missing_values",
        "cif_2016_dataset",
        "fred_md_dataset",
        "hospital_dataset",
        "m4_hourly_dataset",
        "electricity_weekly_dataset",
        "australian_electricity_demand_dataset",
        "tourism_monthly_dataset",
        "tourism_quarterly_dataset",
        "elecdemand_dataset",
        "sunspot_dataset_without_missing_values",
        "wind_4_seconds_dataset",
    ]

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
    print("Total number of test tokens:", test_dataset.total_length())

    print("testing here:", len(test_dataloader), test_dataloader.__len__())
    print("train here:", len(train_dataloader), train_dataloader.__len__())

    return None


if __name__ == "__main__":
    train()
