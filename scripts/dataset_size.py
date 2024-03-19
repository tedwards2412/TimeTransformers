import os
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from tqdm import tqdm
import yaml
import argparse
import json
import numpy as np
import time
import wandb

# This is just until temporary implementation
import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

from data_handling import TimeSeriesDataset, load_datasets
from utils import GradualWarmupScheduler
import Transformer


def train(config):
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    # Accessing the configuration values
    train_split = config["train"]["train_split"]
    max_seq_length = config["train"]["max_seq_length"]
    batch_size = config["train"]["batch_size"]
    test_batch_size = config["train"]["test_batch_size"]
    test_size = config["train"]["test_size"]

    # Datasets
    datasets_to_load = config["datasets"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available() else "cpu"
        else "cpu"
    )
    print(f"Using {device}")
    if device.type == "cuda":
        num_workers = 11
    elif device.type == "mps" or device.type == "cpu":
        num_workers = 6

    print("Number of workers: ", num_workers)

    # First lets download the data and make a data loader
    print("Loading data...")
    (
        training_data_list,
        test_data_list,
        train_masks,
        test_masks,
    ) = load_datasets(datasets_to_load, train_split)

    train_dataset = TimeSeriesDataset(training_data_list, max_seq_length, train_masks)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_dataset = TimeSeriesDataset(
        test_data_list,
        max_seq_length,
        test_masks,
        test=True,
        test_size=test_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    print("Training dataset size: ", train_dataset.__len__())
    print("Test dataset size: ", test_dataset.__len__())

    print("Total number of training tokens:", train_dataset.total_length())
    print("Total number of test tokens:", test_dataset.total_length())

    print("Train batches: ", len(train_dataloader))
    print("Test batches: ", len(test_dataloader))

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script configuration.")
    parser.add_argument(
        "config_file", type=str, help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    train(args.config_file)
