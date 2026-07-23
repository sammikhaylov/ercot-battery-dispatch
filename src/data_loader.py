# Loading CSV / yfinance / ERCOT data
import pandas as pd
from pathlib import Path
from pyarrow import parquet

def load_ercot_data(folder: str) -> pd.DataFrame:
    data = sorted(Path(folder).glob("ercot_*.xlsx"))
    dfs = []

    for d in data:
        print(f"Loading {d}...") # Diagnostic print statement to indicate which file is being loaded
        temp = pd.read_excel(d, sheet_name=None)
        temp = pd.concat(temp, ignore_index=True)
        dfs.append(temp)
        print(type(temp), temp.shape) # Diagnostic print statement to confirm the type and shape of the loaded data
        print(f"Loaded {d}") # Diagnostic print statement to confirm that the file has been loaded successfully
    print(f"Total files loaded: {len(dfs)}") # Diagnostic print statement to indicate the total number of files loaded
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["Settlement Point Name"] == "HB_HOUSTON"].copy()
    df["Delivery Date"] = pd.to_datetime(df["Delivery Date"])
    df["hour_offset"] = df["Delivery Hour"] - 1
    df["minute_offset"] = (df["Delivery Interval"] - 1) * 15
    df["time_stamp"] = (df["Delivery Date"] + pd.to_timedelta(df["hour_offset"], unit='h') + pd.to_timedelta(df["minute_offset"], unit='m'))
    df = df.sort_values("time_stamp").reset_index(drop = True)
    return df

def split_ercot_data(df: pd.DataFrame, train_end: int, test_year: int, backtest_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["time_stamp"].dt.year <= train_end].copy()
    test_df = df[df["time_stamp"].dt.year == test_year].copy()
    backtest_df = df[df["time_stamp"].dt.year == backtest_year].copy()
    return train_df, test_df, backtest_df

def load_load_data(folder: str) -> pd.DataFrame:
    data = sorted(Path(folder).glob("Native_Load_*.xlsx"))
    load_dfs = []

    for d in data:
        print(f"Loading {d}...") # Diagnostic print statement to indicate which file is being loaded
        temp = pd.read_excel(d, sheet_name=None)
        temp = pd.concat(temp, ignore_index=True)
        load_dfs.append(temp)
        print(type(temp), temp.shape) # Diagnostic print statement to confirm the type and shape of the loaded data
        print(f"Loaded {d}") # Diagnostic print statement to confirm that the file has been loaded successfully
    print(f"Total files loaded: {len(load_dfs)}") # Diagnostic print statement to indicate the total number of files loaded
    load = pd.concat(load_dfs, ignore_index=True)
    load = load[["Hour Ending", "ERCOT", "COAST"]]
    time_col = load["Hour Ending"].astype(str).str.replace(" DST", "", regex = False).str.replace(" CST", "", regex = False).str.strip()
    is_24 = time_col.str.contains(r"\b24:00(?::00)?\b", regex = True)
    time_col = time_col.str.replace(r"\b24:00(?::00)?\b", "00:00", regex = True)
    load["Hour Ending"] = pd.to_datetime(time_col, format = "mixed")
    load.loc[is_24, "Hour Ending"] += pd.Timedelta(days=1)
    load = load[["Hour Ending", "ERCOT", "COAST"]].set_index("Hour Ending").sort_index()
    load = load.groupby(level = 0)[["ERCOT", "COAST"]].mean() # Remove Duplicates
    load.index = load.index - pd.Timedelta(minutes = 45)
    load_15min = load.resample("15min").ffill()
    load_15min.index.name = "time_stamp"

    load_15min.to_parquet(
        "data/processed/load_processed.parquet",
        index = False
    )
    return load_15min

def load_weather_data(folder: str) -> pd.DataFrame:
    data = sorted(Path(folder).glob("open-meteo-*.csv"))
    print(f"Loading {data[0]}...") # Diagnostic print statement to indicate which file is being loaded
    weather = pd.read_csv(data[0], skiprows = 3)
    print(type(weather), weather.shape) # Diagnostic print statement to confirm the type and shape of the loaded data
    print(f"Loaded {data[0]}") # Diagnostic print statement to confirm that the file has been loaded successfully
    weather = weather[["time", "temperature_2m (°C)", "dew_point_2m (°C)", "wind_speed_10m (km/h)", "cloud_cover (%)", "shortwave_radiation (W/m²)", "precipitation (mm)"]].copy()
    weather["time"] = pd.to_datetime(weather["time"], utc = True)
    weather["time_stamp"] = weather["time"].dt.tz_convert("America/Chicago").dt.tz_localize(None)
    weather = weather.set_index("time_stamp").sort_index()
    weather = weather.drop(columns = "time")
    weather = weather.groupby(level = 0).mean() # Remove duplicates
    weather = weather.resample("15min").ffill()
    weather = weather.reset_index()
    weather.to_parquet(
        "data/processed/weather_processed.parquet",
        index = False
    )
    return weather