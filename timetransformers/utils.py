import os
from datetime import datetime
from distutils.util import strtobool
import torch
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler

import pandas as pd


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


def convert_df_to_numpy(dfs, train_split=0.8):
    training_data = []
    test_data = []
    train_masks = []
    test_masks = []

    for df in dfs:
        # Select the 'series_value' column from the filtered DataFrame
        selected_series_values = df["series_value"].to_numpy()

        # selected_series_values = selected_series_values.to_numpy()

        # def fill_nans(array):
        #     array = pd.Series(array)
        #     array.ffill(inplace=True)  # Forward fill
        #     array.bfill(inplace=True)
        #     return array.to_numpy()

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

        for i in range(N_data):
            new_data, mask = fill_nans_and_create_mask(
                selected_series_values[i].to_numpy().astype(float)
            )
            new_data_length = new_data.shape[0]

            # Need to append test and train masks
            training_data.append(new_data[: int(train_split * new_data_length)])
            train_masks.append(mask[: int(train_split * new_data_length)])

            test_data.append(new_data[int(train_split * new_data_length) :])
            test_masks.append(mask[int(train_split * new_data_length) :])

    return training_data, test_data, train_masks, test_masks


def add_NOAA_dataset(training_data, test_data, train_masks, test_masks, train_split):
    weather_data = np.load("../data/NOA_data/weather.npz")

    for data_name in tqdm(weather_data.files):
        data = weather_data[data_name]

        for i in range(data.shape[0]):
            current_ts = data[i]
            new_data_length = current_ts.shape[0]
            mask = np.ones(new_data_length)

            # Need to append test and train masks
            training_data.append(current_ts[: int(train_split * new_data_length)])
            train_masks.append(mask[: int(train_split * new_data_length)])

            test_data.append(current_ts[int(train_split * new_data_length) :])
            test_masks.append(mask[int(train_split * new_data_length) :])

    return training_data, test_data, train_masks, test_masks


def PIT(transformer_pred, y_true):
    mean = transformer_pred[:, :, 0].cpu().detach().numpy()
    var = torch.nn.functional.softplus(transformer_pred[:, :, 1])
    std = np.sqrt(var.cpu().detach().numpy())

    U = norm.cdf(
        y_true.cpu().detach().numpy(),
        loc=mean,
        scale=std,
    )
    return U


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_warmup_steps, after_scheduler=None):
        self.total_warmup_steps = total_warmup_steps
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.total_warmup_steps:
            return [
                base_lr * float(self.last_epoch) / self.total_warmup_steps
                for base_lr in self.base_lrs
            ]
        if self.after_scheduler:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                self.finished_warmup = True
            return self.after_scheduler.get_last_lr()
        return self.base_lrs

    def step(self, epoch=None):
        if self.finished_warmup and self.after_scheduler:
            self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
