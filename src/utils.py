# Helper functions
import pandas as pd

def validate_ercot_data(df: pd.DataFrame) -> str:
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame Head:\n{df.head()}")
    print(f"DataFrame Info:\n{df.info()}")
    print(f"DataFrame Description:\n{df.describe()}")
    print(f"DataFrame Null Values:\n{df.isnull().sum()}")
    print(f"DataFrame Columns:\n{df.columns}")
    print(f"DataFrame Index:\n{df.index}")
    print(f"DataFrame Types:\n{df.dtypes}")
    print(f"DataFrame Memory Usage:\n{df.memory_usage(deep=True)}")
    print(f"Time Stamp Range: {df['time_stamp'].min()} to {df['time_stamp'].max()}")
    print(f"Settlement Point Price Range: {df['Settlement Point Price'].min()} to {df['Settlement Point Price'].max()}")
    dupes = df[df["time_stamp"].duplicated(keep=False)]
    print(f"Time Stamp Duplicates: {df['time_stamp'].duplicated().sum()}")
    return dupes

def duplicate_time_stamps(df: pd.DataFrame) -> pd.DataFrame:
    dupes = validate_ercot_data(df)
    duplicate_summary = dupes.groupby("time_stamp")["Settlement Point Price"].agg(["count", "mean", "std", "min", "max"])
    print(f"Duplicate Time Stamps Summary:\n{duplicate_summary}")
    print(f"Duplicate Time Stamps Percentage: {len(duplicate_summary) / len(df) * 100:.4f}%")

def validate_split_data(df: pd.DataFrame) -> str:
    print(df.shape)
    print(df["time_stamp"].min(), df["time_stamp"].max())