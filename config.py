# config.py

"""
Cache Utilization
"""
USE_DATA_CACHE = False
USE_ML_MODEL_CACHE = False

USE_OPTIMIZATION_CACHE = False
USE_RULES_BACKTEST_CACHE = False
USE_RL_CACHE = False

USE_VISUALIZATION_CACHE = False
###

"""
Model Objectives
"""
TARGET = "Settlement Point Price"
TIME_COLUMN = ""

TRAIN_END = 2024
TEST_END = 2025
BACKTEST_START = 2026
###

"""
Feature Engineering
"""
FEATURES = [
    "hour", "day_of_week", "month",
    "lag_1", "lag_4", "lag_12", "lag_24", "lag_96", "lag_672",
    "is_weekend",
    "rolling_mean_4", "rolling_std_4",
    "rolling_mean_12", "rolling_std_12",
    "rolling_mean_24", "rolling_std_24",
    "rolling_mean_48", "rolling_std_48",
    "rolling_mean_96", "rolling_std_96",
    "rolling_mean_192", "rolling_std_192",
    "rolling_mean_672", "rolling_std_672",
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
#    "load_lag_1", "load_lag_4",
    "load_change_1", "load_change_4",
#    "load_pct_change_1", "load_pct_change_4",
#    "load_rolling_mean_12", "load_rolling_mean_96",
    "load_deviation_96", "load_volatility_24",
    "temp_lag_1", "temp_lag_4",
    "temp_change_1", "temp_change_4",
    "temp_rolling_mean_24", "temp_volatility_24",
    "wind_lag_1", "wind_change_4",
    "wind_rolling_mean_24", "wind_volatility_24",
    "solar_lag_1", "solar_change_4",
    "solar_rolling_mean_4", "solar_volatility_24",
    "cloud_lag_1", "cloud_change_4",
    "dew_lag_1", "dew_change_4",
    "precip_lag_1", "heat_index_proxy_1",
]
# Price
PRICE_LAGS = [1, 4, 12, 24, 96, 672] # 15 minute, 1 hour, 3 hour, 6 hour, 1 day, 1 week
PRICE_ROLLING_WINDOWS = [4, 12, 24, 48, 96, 192, 672] # 1 hour, 3 hour, 6 hour, 12 hour, 1 day, 2 day, 1 week
# Load
LOAD_LAGS_AND_CHANGE = [1, 4] # 15 minute, 1 hour
LOAD_ROLLING_WINDOWS = [12, 96] # 3 hour, 1 day
# Weather
WEATHER_LAGS_AND_CHANGE = [1, 4] # 15 minute, 1 hour
###

RAW_DATA_PATH = ""
PROCESSED_DATA_PATH = ""
MODEL_PATH = ""
FIGURE_PATH = ""

"""
ML Model Configurations
"""

# XGBoost
N_ESTIMATORS_XGB = 500
LEARNING_RATE_XGB = 0.05
MAX_DEPTH_XGB = 6
SUBSAMPLE_XGB = 0.8
COLSAMPLE_BYTREE_XGB = 0.8
OBJECTIVE_XGB = "reg:squarederror"
RANDOM_STATE_XGB = 42
N_JOBS_XGB = -1

# LightGBM
N_ESTIMATORS_LGBM = 500
LEARNING_RATE_LGBM = 0.05
MAX_DEPTH_LGBM = -1
NUM_LEAVES_LGBM = 31
SUBSAMPLE_LGBM = 0.8
COLSAMPLE_BYTREE_LGBM = 0.8
RANDOM_STATE_LGBM = 42
N_JOBS_LGBM = -1

# Hyperparameter config for tuned XGBoost & LightGBM
RANDOM_STATE_CV = 42
CV_SPLITS = 5
###

"""
Battery Assumptions
"""
BATTERY_CAPACITY_MWH = 100.0

MAX_CHARGE_MW = 25.0
MAX_DISCHARGE_MW = 25.0

CHARGE_EFFICIENCY = 0.95 # %
DISCHARGE_EFFICIENCY = 0.95 # %

INITIAL_SOC_MWH = 50.0
MINIMUM_SOC_MWH = 10.0

INTERVAL_HOURS = 0.25
DEGRADATION_COST_PER_MWH = 5.0 # $ USD
TRANSACTION_COST_PER_MWH = 0.50 # $ USD

"""
Rules-Based Backtest Benchmark
"""
THRESHOLD_WINDOW = 672 # Trailing 7 days
LOWER_QUANTILE = 0.25
UPPER_QUANTILE = 0.75
MINIMUM_HISTORY = 96 # Requirement for initiating rules