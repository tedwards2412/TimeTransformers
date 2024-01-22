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
    train_split = 0.8
    max_seq_length = 1024
    # batch_size = 512
    # test_batch_size = 1024

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
    # print(len(datasets_to_load.keys()))
    # quit()
    dfs = download_data(datasets_to_load)

    training_data_list, test_data_list, train_masks, test_masks = convert_df_to_numpy(
        dfs, train_split
    )

    train_dataset = TimeSeriesDataset(training_data_list, max_seq_length, train_masks)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TimeSeriesDataset(test_data_list, max_seq_length, test_masks)
    # test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    print("Training dataset size: ", train_dataset.__len__())
    print("Test dataset size: ", test_dataset.__len__())

    print("Total number of training tokens:", train_dataset.total_length())
    print("Total number of test tokens:", test_dataset.total_length())

    return None


if __name__ == "__main__":
    train()
