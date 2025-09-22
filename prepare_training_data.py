import pandas as pd
import numpy as np
import torch

def load_data(data_path):
    # data_path = config.PROCESSED_DATA_DIR
    df = pd.read_csv(data_path)
    return df

def df_eda(data):
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    elif isinstance(data, torch.Tensor):
        df = pd.DataFrame(data.numpy())
    # Perform exploratory data analysis (EDA) on the DataFrame
    print("DataFrame Overview:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nStatistical Summary:")
    print(df.describe())

def fill_na(data):
    # Fill NA with zeros
    if isinstance(data, pd.DataFrame):
        return data.fillna(0.0)
    elif isinstance(data, np.ndarray):
        return np.nan_to_num(data, nan=0.0)
    elif isinstance(data, torch.Tensor):
        return data.fillna(0.0)
    return data

def create_mask(data):
    # Create a mask for missing values
    if isinstance(data, pd.DataFrame):
        return ~data.isnull()
    elif isinstance(data, np.ndarray):
        return ~np.isnan(data)
    elif isinstance(data, torch.Tensor):
        return ~data.isnan()
    return None

def prepare_data(config):
    data = load_data(config.PROCESSED_DATA_DIR / "processed_data.csv")
    mask = create_mask(data)
    data = fill_na(data)
    