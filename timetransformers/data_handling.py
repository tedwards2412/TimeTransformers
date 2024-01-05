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
    if mean == 0 or std == 0:
        return data
    else:
        return (data - mean) / std


class TimeSeriesDataset(Dataset):
    def __init__(self, data, max_sequence_length):
        self.max_sequence_length = max_sequence_length
        self.means = np.array([np.mean(data[i]) for i in range(len(data))])
        self.std = np.array([np.std(data[i]) for i in range(len(data))])

        self.data = [
            normalize_data(data[i], self.means[i], self.std[i])
            for i in range(len(data))
        ]
        self.probs = (
            np.array([len(self.data[i]) for i in range(len(self.data))])
            / self.total_length()
        )

    def __len__(self):
        l = 0
        for i in range(len(self.data)):
            l += len(self.data[i])
        return int(l / self.max_sequence_length)

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
            mask = torch.ones_like(train_series, dtype=torch.bool)

            return (
                train_series,
                true_series,
                mask.squeeze(-1),
            )

        else:
            train_series = torch.tensor(
                series[:-1],
                dtype=torch.float32,
            )
            true_series = torch.tensor(series[1:], dtype=torch.float32)

            mask = torch.ones(len(train_series), dtype=torch.bool)

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
    os.system(f"zenodo_get {dataset_id}")
    os.system(f"mv {dataset_name}.zip ../data/{dataset_name}.zip")
    print(f"Downloaded {dataset_name}.tsf")

    # Unzip the dataset
    os.system(f"unzip -o ../data/{dataset_name}.zip -d ../data/")
    print(f"Unzipped {dataset_name}.tsf")

    # Remove the zip file
    os.system(f"rm ../data/{dataset_name}.zip")
    os.system(f"rm md5sums.txt")

    # Convert the tsf file to a pandas dataframe
    return convert_tsf_to_dataframe(f"../data/{dataset_name}.tsf")[0]


def download_data(dataset_dict):
    df_list = []
    for dataset_name, dataset_id in dataset_dict.items():
        df_list.append(download_single_datafile(dataset_name, dataset_id))

    return df_list
