# Creating lags / rolling features
import pandas as pd
import numpy as np
from config import PRICE_LAGS, PRICE_ROLLING_WINDOWS, LOAD_LAGS_AND_CHANGE, LOAD_ROLLING_WINDOWS, WEATHER_LAGS_AND_CHANGE

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])

    df["hour"] = df["time_stamp"].dt.hour
    df["day_of_week"] = df["time_stamp"].dt.day_of_week
    df["month"] = df["time_stamp"].dt.month
    
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # cyclical time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df

def create_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    lags = PRICE_LAGS
    rolling_windows = PRICE_ROLLING_WINDOWS
    df = df.copy()
    for l in lags:
        df[f"lag_{l}"] = df["Settlement Point Price"].shift(l)

    price = df["Settlement Point Price"].shift(1) # Prevent data leakage
    for rw in rolling_windows:
        df[f"rolling_mean_{rw}"] = price.rolling(rw).mean()
        df[f"rolling_std_{rw}"] = price.rolling(rw).std()
    return df

def create_load_features(df: pd.DataFrame) -> pd.DataFrame:
    lags_change = LOAD_LAGS_AND_CHANGE
    rolling_windows = LOAD_ROLLING_WINDOWS
    df = df.copy()
    for l in lags_change:
        df[f"load_lag_{l}"] = df["ERCOT"].shift(l) # depends on ERCOT system, not COAST
    
    df["known_load"] = df["ERCOT"].shift(1) # Prevent data leakage
    for rw in rolling_windows:
        df[f"load_rolling_mean_{rw}"] = df["known_load"].rolling(rw).mean()
    
    for c in lags_change:
        df[f"load_change_{c}"] = df["known_load"].diff(c)
    
    for p in lags_change:
        df[f"load_pct_change_{p}"] = df["known_load"].pct_change(p)
    
    df["load_deviation_96"] = df["known_load"] - df["load_rolling_mean_96"]
    df["load_volatility_24"] = df["known_load"].rolling(24).std()
    return df

def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    lags_change = WEATHER_LAGS_AND_CHANGE
    df = df.copy()
    for l in lags_change:
        df[f"temp_lag_{l}"] = df["temperature_2m (°C)"].shift(l)
    
    for c in lags_change:
        df[f"temp_change_{c}"] = df["temperature_2m (°C)"].diff(c)
    
    df["temp_rolling_mean_24"] = df["temperature_2m (°C)"].rolling(24).mean()
    df["temp_volatility_24"] = df["temperature_2m (°C)"].rolling(24).std()

    df["wind_lag_1"] = df["wind_speed_10m (km/h)"].shift(1)
    df["wind_change_4"] = df["wind_speed_10m (km/h)"].diff(4)
    df["wind_rolling_mean_24"] = df["wind_speed_10m (km/h)"].rolling(24).mean()
    df["wind_volatility_24"] = df["wind_speed_10m (km/h)"].rolling(24).std()

    df["solar_lag_1"] = df["shortwave_radiation (W/m²)"].shift(1)
    df["solar_change_4"] = df["shortwave_radiation (W/m²)"].diff(4)
    df["solar_rolling_mean_4"] = df["shortwave_radiation (W/m²)"].rolling(4).mean()
    df["solar_volatility_24"] = df["shortwave_radiation (W/m²)"].rolling(24).std()

    df["cloud_lag_1"] = df["cloud_cover (%)"].shift(1)
    df["cloud_change_4"] = df["cloud_cover (%)"].diff(4)

    df["dew_lag_1"] = df["dew_point_2m (°C)"].shift(1)
    df["dew_change_4"] = df["dew_point_2m (°C)"].diff(4)

    df["precip_lag_1"] = df["precipitation (mm)"].shift(1)

    # combinations
    df["heat_index_proxy_1"] = df["temperature_2m (°C)"].shift(1) + 0.2 * df["dew_point_2m (°C)"].shift(1)
    return df