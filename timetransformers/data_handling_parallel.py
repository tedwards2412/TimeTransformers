import numpy as np
import os
from datetime import datetime
from distutils.util import strtobool
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import random
import pandas as pd
import os
import glob


data_path = "/home/tedwar42/data_tedwar42/final"
# data_path = ""
# print("Data path: ", data_path)


def normalize_data(data, mean, std):
    if std == 0:
        return data - mean
    else:
        return (data - mean) / std


class TimeSeriesDataset(Dataset):
    def __init__(
        self, data, max_sequence_length, miss_vals_mask, test=False, test_size=0.2
    ):
        self.miss_vals_mask = miss_vals_mask
        self.max_sequence_length = max_sequence_length
        self.means = np.array(
            [
                np.mean(data[i][np.array(self.miss_vals_mask[i], dtype=bool)])
                for i in tqdm(range(len(data)), desc="Calculating means")
            ]
        )
        self.std = np.array(
            [
                np.std(data[i][np.array(self.miss_vals_mask[i], dtype=bool)])
                for i in tqdm(range(len(data)), desc="Calculating std")
            ]
        )

        self.data = [
            normalize_data(data[i], self.means[i], self.std[i])
            for i in tqdm(range(len(data)), desc="Normalizing Data")
        ]
        self.probs = (
            np.array(
                [
                    len(self.data[i])
                    for i in tqdm(range(len(self.data)), desc="Calculating Probs")
                ]
            )
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
    tsf_file_path = data_path + f"/monash/{dataset_name}.tsf"
    zip_file_path = data_path + f"/monash/{dataset_name}.zip"

    # Check if the dataset already exists
    if not os.path.exists(tsf_file_path):
        # Download the dataset if it doesn't exist
        os.system(f"zenodo_get {dataset_id}")
        os.system(f"mv {dataset_name}.zip {zip_file_path}")
        print(f"Downloaded {dataset_name}.zip")

        # Unzip the dataset
        os.system(f"unzip -o {zip_file_path} -d {tsf_file_path}")
        print(f"Unzipped {dataset_name}.zip")

        # Remove the zip file
        os.system(f"rm {zip_file_path}")
        os.system(f"rm md5sums.txt")
    else:
        print(f"{dataset_name}.tsf already exists. Skipping download.")

    # Convert the tsf file to a pandas dataframe
    return convert_tsf_to_dataframe(data_path + f"/monash/{dataset_name}.tsf")[0]


def download_data(dataset_names):
    df_list = []
    for dataset_name in dataset_names:
        dataset_id = monash_dataset_dict[dataset_name]
        df_list.append(download_single_datafile(dataset_name, dataset_id))

    return df_list


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def load_datasets(datasets, train_split, rank, world_size):
    np.random.seet(42)
    if datasets["monash"]:
        print("Adding Monash data...")
        dfs = download_data(datasets["monash"])

        (
            training_data_list,
            test_data_list,
            train_masks,
            test_masks,
        ) = convert_df_to_numpy(dfs, rank, world_size, train_split)

    if datasets["weather"]:
        print("Adding weather data...")
        for data in datasets["weather"]:
            path = weather_paths[data]
            (
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
            ) = add_weather_dataset(
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
                train_split,
                path,
                rank,
                world_size,
            )
    if datasets["traffic"]:
        print("Adding traffic data...")
        if "CA" in datasets["traffic"]:
            print("Adding California traffic data...")
            path = traffic_paths["CA"]
            (
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
            ) = add_ca_traffic_dataset(
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
                train_split,
                path,
                rank,
                world_size,
            )
    if datasets["finance"]:
        print("Adding finance data...")
        if "yahoo" in datasets["finance"]:
            print("Adding Yahoo Finance data...")
            (
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
            ) = add_yahoo_dataset(
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
                train_split,
                rank,
                world_size,
            )
    if datasets["science"]:
        print("Adding science data...")
        if "ZTF" in datasets["science"]:
            (
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
            ) = add_ZTF_dataset(
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
                train_split,
                rank,
                world_size,
            )
    if datasets["audio"]:
        print("Adding audio data...")
        if "arabic" in datasets["audio"]:
            (
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
            ) = add_arabic_audio_dataset(
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
                train_split,
                rank,
                world_size,
            )
        if "commands" in datasets["audio"]:
            (
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
            ) = add_command_audio_dataset(
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
                train_split,
                rank,
                world_size,
            )
        if "birds" in datasets["audio"]:
            (
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
            ) = add_bird_audio_dataset(
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
                train_split,
                rank,
                world_size,
            )
    if datasets["energy"]:
        print("Adding energy data...")
        if "buildingbench" in datasets["energy"]:
            (
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
            ) = add_buildingbench_dataset(
                training_data_list,
                test_data_list,
                train_masks,
                test_masks,
                train_split,
                rank,
                world_size,
            )

    return training_data_list, test_data_list, train_masks, test_masks


def convert_df_to_numpy(dfs, rank, world_size, train_split=0.8):
    training_data = []
    test_data = []
    train_masks = []
    test_masks = []

    for df in dfs:
        # Select the 'series_value' column from the filtered DataFrame
        selected_series_values = df["series_value"].to_numpy()

        def fill_nans_and_create_mask(array):
            # Convert input array to a Pandas Series
            series = pd.Series(array)

            # Create a mask: True where NaN, False otherwise
            mask_ = ~series.isna()

            # Fill NaNs with zero
            series.fillna(0.0, inplace=True)

            array_filled = series.to_numpy()
            mask_array = mask_.to_numpy()

            return array_filled, mask_array

        N_data = selected_series_values.shape[0]
        samples_per_proc = N_data // world_size
        start_index = rank * samples_per_proc
        end_index = N_data if rank == world_size - 1 else (rank + 1) * samples_per_proc

        index_arange = np.arange(start_index, end_index)
        n_train_data = int(len(index_arange) * train_split)
        train_indices = set(np.random.choice(index_arange, n_train_data, replace=False))

        for i in tqdm(index_arange):
            new_data, mask = fill_nans_and_create_mask(
                selected_series_values[i].to_numpy().astype(float)
            )

            if i in train_indices:
                training_data.append(new_data)
                train_masks.append(mask)
            else:
                test_data.append(new_data)
                test_masks.append(mask)

    return training_data, test_data, train_masks, test_masks


def add_buildingbench_dataset(
    training_data, test_data, train_masks, test_masks, train_split, rank, world_size
):
    buildingbench_files = glob.glob(energy_paths["buildingbench"] + "*.npy")
    rnd_indices = np.random.choice(len(buildingbench_files), 40, replace=False)

    for rnd in tqdm(rnd_indices):
        buildingbench_data = np.load(buildingbench_files[rnd])
        N_data = buildingbench_data.shape[0]
        samples_per_proc = N_data // world_size
        start_index = rank * samples_per_proc
        end_index = N_data if rank == world_size - 1 else (rank + 1) * samples_per_proc

        index_arange = np.arange(start_index, end_index)
        n_train_data = int(len(index_arange) * train_split)
        train_indices = set(np.random.choice(index_arange, n_train_data, replace=False))

        for i in tqdm(index_arange):
            current_ts = buildingbench_data[i]
            new_data_length = current_ts.shape[0]
            mask = np.ones(new_data_length)

            # Need to append test and train masks
            if i in train_indices:
                training_data.append(current_ts)
                train_masks.append(mask)
            else:
                test_data.append(current_ts)
                test_masks.append(mask)

    return training_data, test_data, train_masks, test_masks


def add_weather_dataset(
    training_data,
    test_data,
    train_masks,
    test_masks,
    train_split,
    path,
    rank,
    world_size,
):
    weather_data = np.load(path)

    for data_name in tqdm(weather_data.files):
        data = weather_data[data_name]
        N_data = data.shape[0]
        samples_per_proc = N_data // world_size
        start_index = rank * samples_per_proc
        end_index = N_data if rank == world_size - 1 else (rank + 1) * samples_per_proc

        index_arange = np.arange(start_index, end_index)
        n_train_data = int(len(index_arange) * train_split)
        train_indices = set(np.random.choice(index_arange, n_train_data, replace=False))

        for i in tqdm(index_arange):
            current_ts = data[i]
            new_data_length = current_ts.shape[0]
            mask = np.ones(new_data_length)

            # Need to append test and train masks
            if i in train_indices:
                training_data.append(current_ts)
                train_masks.append(mask)
            else:
                test_data.append(current_ts)
                test_masks.append(mask)

    return training_data, test_data, train_masks, test_masks


def add_ca_traffic_dataset(
    training_data,
    test_data,
    train_masks,
    test_masks,
    train_split,
    path,
    rank,
    world_size,
):
    traffic_data = np.load(path)
    data = traffic_data["arr_0"]

    N_data = data.shape[0]
    samples_per_proc = N_data // world_size
    start_index = rank * samples_per_proc
    end_index = N_data if rank == world_size - 1 else (rank + 1) * samples_per_proc

    index_arange = np.arange(start_index, end_index)
    n_train_data = int(len(index_arange) * train_split)
    train_indices = set(np.random.choice(index_arange, n_train_data, replace=False))

    for i in tqdm(index_arange):
        current_ts = data[i]
        new_data_length = current_ts.shape[0]
        mask = ~np.isnan(current_ts)
        current_ts[~mask] = 0.0

        # Need to append test and train masks
        if i in train_indices:
            training_data.append(current_ts)
            train_masks.append(mask)
        else:
            test_data.append(current_ts)
            test_masks.append(mask)

    return training_data, test_data, train_masks, test_masks


def add_yahoo_dataset(
    training_data, test_data, train_masks, test_masks, train_split, rank, world_size
):
    stock_returns = np.load(finance_paths["yahoo_returns"], allow_pickle=True)
    stock_volume = np.load(finance_paths["yahoo_volume"], allow_pickle=True)

    print("Adding stock returns...")
    returns = stock_returns["cleaned_stock_returns"]
    N_data = returns.shape[0]
    samples_per_proc = N_data // world_size
    start_index = rank * samples_per_proc
    end_index = N_data if rank == world_size - 1 else (rank + 1) * samples_per_proc

    index_arange = np.arange(start_index, end_index)
    n_train_data = int(len(index_arange) * train_split)
    train_indices = set(np.random.choice(index_arange, n_train_data, replace=False))

    for i in tqdm(index_arange):
        current_ts = returns[i]
        new_data_length = current_ts.shape[0]
        mask = np.ones(new_data_length)

        # Need to append test and train masks
        if i in train_indices:
            training_data.append(current_ts)
            train_masks.append(mask)
        else:
            test_data.append(current_ts)
            test_masks.append(mask)

    print("Adding stock volume...")
    volume = stock_volume["cleaned_stock_volume"]
    N_data = volume.shape[0]
    samples_per_proc = N_data // world_size
    start_index = rank * samples_per_proc
    end_index = N_data if rank == world_size - 1 else (rank + 1) * samples_per_proc

    index_arange = np.arange(start_index, end_index)
    n_train_data = int(len(index_arange) * train_split)
    train_indices = set(np.random.choice(index_arange, n_train_data, replace=False))

    for i in tqdm(index_arange):
        current_ts = np.log10(1 + volume[i])
        new_data_length = current_ts.shape[0]
        mask = np.ones(new_data_length)

        # Need to append test and train masks
        if i in train_indices:
            training_data.append(current_ts)
            train_masks.append(mask)
        else:
            test_data.append(current_ts)
            test_masks.append(mask)

    return training_data, test_data, train_masks, test_masks


def add_command_audio_dataset(
    training_data, test_data, train_masks, test_masks, train_split, rank, world_size
):
    speech_commands = np.load(audio_paths["commands"])

    print("Adding speech command audio...")
    speech_commands = speech_commands["audio_array"].astype(np.float32)
    N_data = speech_commands.shape[0]
    samples_per_proc = N_data // world_size
    start_index = rank * samples_per_proc
    end_index = N_data if rank == world_size - 1 else (rank + 1) * samples_per_proc

    index_arange = np.arange(start_index, end_index)
    n_train_data = int(len(index_arange) * train_split)
    train_indices = set(np.random.choice(index_arange, n_train_data, replace=False))

    for i in tqdm(index_arange):
        current_ts = speech_commands[i]
        new_data_length = current_ts.shape[0]
        mask = np.ones(new_data_length)

        # Need to append test and train masks
        if i in train_indices:
            training_data.append(current_ts)
            train_masks.append(mask)
        else:
            test_data.append(current_ts)
            test_masks.append(mask)
    return training_data, test_data, train_masks, test_masks


def add_arabic_audio_dataset(
    training_data, test_data, train_masks, test_masks, train_split, rank, world_size
):
    arabic_audio = np.load(audio_paths["arabic"])

    print("Adding arabic audio...")
    arabic_audio = arabic_audio["downsampled_audio_padded"].astype(np.float32)
    N_data = arabic_audio.shape[0]
    samples_per_proc = N_data // world_size
    start_index = rank * samples_per_proc
    end_index = N_data if rank == world_size - 1 else (rank + 1) * samples_per_proc

    index_arange = np.arange(start_index, end_index)
    n_train_data = int(len(index_arange) * train_split)
    train_indices = set(np.random.choice(index_arange, n_train_data, replace=False))

    for i in tqdm(index_arange):
        ts_length = int(arabic_audio[i, -1])
        current_ts = arabic_audio[i, :ts_length]
        new_data_length = current_ts.shape[0]
        mask = np.ones(new_data_length)

        # Need to append test and train masks
        if i in train_indices:
            training_data.append(current_ts)
            train_masks.append(mask)
        else:
            test_data.append(current_ts)
            test_masks.append(mask)
    return training_data, test_data, train_masks, test_masks


def add_bird_audio_dataset(
    training_data, test_data, train_masks, test_masks, train_split, rank, world_size
):
    bird_audio = np.load(audio_paths["birds"])

    print("Adding bird audio...")
    bird_audio = bird_audio["downsampled_audio_padded"].astype(np.float32)
    N_data = bird_audio.shape[0]
    samples_per_proc = N_data // world_size
    start_index = rank * samples_per_proc
    end_index = N_data if rank == world_size - 1 else (rank + 1) * samples_per_proc

    index_arange = np.arange(start_index, end_index)
    n_train_data = int(len(index_arange) * train_split)
    train_indices = set(np.random.choice(index_arange, n_train_data, replace=False))

    for i in tqdm(index_arange):
        ts_length = int(bird_audio[i, -1])
        current_ts = bird_audio[i, :ts_length]
        new_data_length = current_ts.shape[0]
        mask = np.ones(new_data_length)

        # Need to append test and train masks
        if i in train_indices:
            training_data.append(current_ts)
            train_masks.append(mask)
        else:
            test_data.append(current_ts)
            test_masks.append(mask)
    return training_data, test_data, train_masks, test_masks


traffic_paths = {"CA": data_path + "/traffic/processed_ca_traffic_data.npz"}

energy_paths = {
    "buildingbench": data_path + "/energy/",
}

audio_paths = {
    "arabic": data_path + "/audio/arabic_speech_corpus/arabic_speech.npz",
    "commands": data_path + "/audio/speech_commands/speech_commands.npz",
    "birds": data_path + "/audio/bird_data/bird_audio.npz",
}

science_paths = {"ZTF": data_path + "/science/ZTF_supernova/light_curves.npz"}

finance_paths = {
    "yahoo_returns": data_path + "/finance/yahoo/cleaned_stock_returns.npz",
    "yahoo_volume": data_path + "/finance/yahoo/cleaned_stock_volume.npz",
}

weather_paths = {
    "NOAA_dataset": data_path + "/weather/NOAA/NOAA_weather.npz",
    "ERA5_dataset": data_path + "/weather/ERA5/ERA5.npz",
}

monash_dataset_dict = {
    # Finance
    "nn5_weekly_dataset": "10.5281/zenodo.4656125",
    "nn5_daily_dataset_with_missing_values": "10.5281/zenodo.4656110",
    "cif_2016_dataset": "10.5281/zenodo.4656042",
    "fred_md_dataset": "10.5281/zenodo.4654833",
    "dominick_dataset": "10.5281/zenodo.4654802",
    # Health
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
    # Weather
    "oikolab_weather_dataset": "10.5281/zenodo.5184708",
    "sunspot_dataset_with_missing_values": "10.5281/zenodo.4654773",
    "solar_4_seconds_dataset": "10.5281/zenodo.4656027",
    "wind_4_seconds_dataset": "10.5281/zenodo.4656032",
    "weather_dataset": "10.5281/zenodo.4654822",
    "solar_weekly_dataset": "10.5281/zenodo.4656151",
    "solar_10_minutes_dataset": "10.5281/zenodo.4656144",
    "saugeenday_dataset": "10.5281/zenodo.4656058",
    "wind_farms_minutely_dataset_with_missing_values": "10.5281/zenodo.4654909",
    # Traffic
    "pedestrian_counts_dataset": "10.5281/zenodo.4656626",
    "traffic_weekly_dataset": "10.5281/zenodo.4656135",
    "traffic_hourly_dataset": "10.5281/zenodo.4656132",
    "kaggle_web_traffic_dataset_with_missing_values": "10.5281/zenodo.4656080",
    "london_smart_meters_dataset_with_missing_values": "10.5281/zenodo.4656072",
}