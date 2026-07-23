from src.data_loader import (
    load_ercot_data, 
    split_ercot_data, 
    load_load_data, 
    load_weather_data,
    )
from src.features import (
    create_time_features, 
    create_lag_and_rolling_features, 
    create_load_features,
    create_weather_features,
    )
from src.train import (
    prepare_xy, 
    train_linear_model, 
    train_random_forest, 
    train_xgboost, 
    train_lgbm, 
    tune_xgboost, 
    tune_lightgbm,
    )
from src.evaluate import (
    evaluate_model, get_predictions, select_best_model
    )
from src.visualize import (
    plot_data_split_timeframe,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison,
    plot_monthly_performance,
    plot_residual_analysis,
    plot_permutation_importance,
    plot_shap_summary,
    plot_battery_dispatch_day,
    plot_cumulative_optimization_profit,
    plot_strategy_performance_comparison,
    plot_ppo_action_price_distribution,
    plot_ppo_soc_heatmap,
    plot_forecast_horizon_performance_comparison,
    plot_forecast_horizon_cumulative_profit,
    plot_battery_dispatch_duration_curve,

    )
from src.optimization import (
    optimize_backtest, 
    calculate_optimization_metrics, 
    optimize_backtest_rolling, 
    optimize_backtest_perfect_foresight
    )
from src.rules_based import rules_based_backtest
from src.utils import (
    validate_ercot_data, 
    duplicate_time_stamps, 
    validate_split_data,
    )
from config import (
    USE_DATA_CACHE, 
    USE_ML_MODEL_CACHE, 
    USE_VISUALIZATION_CACHE, 
    USE_OPTIMIZATION_CACHE, 
    USE_RULES_BACKTEST_CACHE, 
    USE_RL_CACHE,
    TARGET, FEATURES, 
    TRAIN_END, TEST_END, BACKTEST_START,
    BATTERY_CAPACITY_MWH, INTERVAL_HOURS,
    )

from pyarrow import parquet
import joblib
import json
import pandas as pd

import time
#import faulthandler
#faulthandler.dump_traceback_later(30, exit = True)

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.reinforcement_learning import (
    BatteryTradingEnv,
    train_ppo_agent,
    evaluate_ppo_agent,
    )
"""
IMPORT DATA AND FORMAT DATAFRAMES
"""
if USE_DATA_CACHE: # Cache framework to only pre-process data once
    df = pd.read_parquet(
        "data/processed/ercot_processed.parquet"
    )
else:
    df = load_ercot_data(
        "data/raw"
        )
    load = load_load_data(
        "data/raw"
        )
    weather = load_weather_data(
        "data/raw"
    )
    df = df.merge(load, on = "time_stamp", how = "left")
    df = df.merge(weather, on = "time_stamp", how = "left")
    df = create_time_features(df)
    df = create_lag_and_rolling_features(df)
    df = create_load_features(df)
    df = create_weather_features(df)
    df = df.dropna().reset_index(drop = True)

    df.to_parquet(
        "data/processed/ercot_processed.parquet",
        index = False
    )

train_df, test_df, backtest_df = split_ercot_data(df, train_end = TRAIN_END, test_year = TEST_END, backtest_year = BACKTEST_START)

# Validation of df's (total, train, test, backtest)
#validate_ercot_data(df) # Diagnostic print statements to validate the loaded and processed data
#duplicate_time_stamps(df) # Diagnostic print statements to check for duplicate time stamps
#print(f"DataFrame Row Length: {len(df.index)}") # Diagnostic print statement to indicate the number of rows in the DataFrame, Expected: 192,380 data points
#validate_split_data(train_df) # Diagnostic print statements for shape and min max
#validate_split_data(test_df) # Diagnostic print statements for shape and min max
#validate_split_data(backtest_df) # Diagnostic print statements for shape and min max
"""
TRAIN ML MODELS
"""
if USE_ML_MODEL_CACHE == False:
    # Integrate ML Models
    X_train, y_train = prepare_xy(train_df, FEATURES, TARGET)
    X_test, y_test = prepare_xy(test_df, FEATURES, TARGET)
    joblib.dump(X_train, "outputs/cache_variables/X_train.pkl")
    joblib.dump(y_train, "outputs/cache_variables/y_train.pkl")
    joblib.dump(X_test, "outputs/cache_variables/X_test.pkl")
    joblib.dump(y_test, "outputs/cache_variables/y_test.pkl")
    linear = train_linear_model(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)
    lgbm = train_lgbm(X_train, y_train)
    xgb_tuned = tune_xgboost(X_train, y_train) # Hyperparameterized
    lgbm_tuned = tune_lightgbm(X_train, y_train) # Hyperparameterized

    linear_metrics = evaluate_model(linear, X_test, y_test)
    rf_metrics = evaluate_model(rf, X_test, y_test)
    xgboost_metrics = evaluate_model(xgb, X_test, y_test)
    lgbm_metrics = evaluate_model(lgbm, X_test, y_test)
    xgb_tuned_metrics = evaluate_model(xgb_tuned, X_test, y_test)
    lgbm_tuned_metrics = evaluate_model(lgbm_tuned, X_test, y_test)

    # Frames for models & metrics
    models = {
        "Linear Regression": linear,
        "Random Forest": rf,
        "XGBoost": xgb,
        "Tuned XGBoost": xgb_tuned,
        "LightGBM": lgbm,
        "Tuned LightGBM": lgbm_tuned
    }
    metrics = {
        "Linear Regression": linear_metrics,
        "Random Forest": rf_metrics,
        "XGBoost": xgboost_metrics,
        "Tuned XGBoost": xgb_tuned_metrics,
        "LightGBM": lgbm_metrics,
        "Tuned LightGBM": lgbm_tuned_metrics
    }

    best_model_name, best_model, results = select_best_model(
        models = models,
        metrics = metrics,
        metric = "RMSE",
        lower_is_better = True
    )

    print(results)
    results.to_csv("outputs/model_comparison.csv", index = False)
    joblib.dump(results, "outputs/cache_variables/results.pkl")
    print("\nBest model:", best_model_name)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(best_model_name, "outputs/cache_variables/best_model_name.pkl")
    best_predictions = get_predictions(best_model, X_test, y_test, test_df["time_stamp"]) # Save primary ML model for predictions
    best_predictions.to_csv("outputs/predictions/best_model_predictions.csv", index = False)
    best_predictions.to_parquet("outputs/predictions/best_model_predictions_2025.parquet", index = False)
    joblib.dump(best_predictions, "outputs/predictions/best_model_predictions_2025.pkl")
if USE_ML_MODEL_CACHE == True:
    best_model = joblib.load("models/best_model.pkl")
    best_model_name = joblib.load("outputs/cache_variables/best_model_name.pkl")
    best_predictions = joblib.load("outputs/predictions/best_model_predictions_2025.pkl")
    X_train = joblib.load("outputs/cache_variables/X_train.pkl")
    y_train = joblib.load("outputs/cache_variables/y_train.pkl")
    X_test = joblib.load("outputs/cache_variables/X_test.pkl")
    y_test = joblib.load("outputs/cache_variables/y_test.pkl")
    results = joblib.load("outputs/cache_variables/results.pkl")


# Apply model to 2026 backtest data
X_backtest, y_backtest = prepare_xy(backtest_df, FEATURES, TARGET)
backtest_predictions = get_predictions(best_model, X_backtest, y_backtest, backtest_df["time_stamp"])
backtest_predictions.to_parquet("outputs/predictions/best_model_predictions_2026.parquet", index = False)
#print(backtest_predictions.columns.tolist())

"""
OPTIMIZATION CVXPY STRATEGY
"""
if USE_OPTIMIZATION_CACHE == False:
    optimized_dispatch = optimize_backtest_rolling(backtest_predictions) # 4 Hour Horizon
    day_optimized_dispatch = optimize_backtest(backtest_predictions) # 24 Hour Horizon
    perfect_optimized_dispatch = optimize_backtest_perfect_foresight(backtest_predictions) # Perfect foresight

    optimization_summary, optimization_daily = calculate_optimization_metrics(optimized_dispatch)
    perfect_optimization_summary, perfect_optimization_daily = calculate_optimization_metrics(perfect_optimized_dispatch)
    day_optimization_summary, day_optimization_daily = calculate_optimization_metrics(day_optimized_dispatch)

    optimized_dispatch.to_parquet("outputs/optimization/optimized_dispatch_2026.parquet", index = False)
    perfect_optimized_dispatch.to_parquet("outputs/optimization/optimized_dispatch_2026_perfect.parquet", index = False)
    day_optimized_dispatch.to_parquet("outputs/optimization/optimized_dispatch_2026_24.parquet")

    joblib.dump(optimization_summary, "outputs/optimization/forecast_cvxpy_summary.pkl")
    joblib.dump(perfect_optimization_summary, "outputs/optimization/perfect_forecast_cvxpy_summary.pkl")
    joblib.dump(day_optimization_summary, "outputs/optimization/24_forecast_cvxpy_summary.pkl")
    
    optimization_daily.to_parquet("outputs/optimization/forecast_cvxpy_daily_metrics.parquet", index = False)
    perfect_optimization_daily.to_parquet("outputs/optimization/perfect_forecast_cvxpy_daily_metrics.parquet", index = False)
    day_optimization_daily.to_parquet("outputs/optimization/24_forecast_cvxpy_daily_metrics.parquet", index = False)
    with open(
        "outputs/optimization/forecast_cvxpy_summary.json",
        "w"
    ) as f:
        json.dump(optimization_summary, f, indent = 4)
    with open(
        "outputs/optimization/perfect_forecast_cvxpy_summary.json",
        "w"
    ) as f:
        json.dump(perfect_optimization_summary, f, indent = 4)
    with open(
        "outputs/optimization/24_forecast_cvxpy_summary.json",
        "w"
    ) as f:
        json.dump(day_optimization_summary, f, indent = 4)
    #first_date = optimized_dispatch["time_stamp"].dt.date.min()
    #first_day = optimized_dispatch[optimized_dispatch["time_stamp"].dt.date == first_date]
    #print(first_day.to_string())

    #simultaneous = first_day[(first_day["Charge MW"] > 1e-6) & (first_day["Discharge MW"] > 1e-6)]
    #print("Simultaneous charge/discharge intervals (day):", len(simultaneous))
    #print(simultaneous)

else:
    # Optimization - 4HR, PERFECT, 24HR
    optimized_dispatch = pd.read_parquet("outputs/optimization/optimized_dispatch_2026.parquet")
    optimization_summary = joblib.load("outputs/optimization/forecast_cvxpy_summary.pkl")
    optimization_daily = pd.read_parquet("outputs/optimization/forecast_cvxpy_daily_metrics.parquet")

    perfect_optimized_dispatch = pd.read_parquet("outputs/optimization/optimized_dispatch_2026_perfect.parquet")
    perfect_optimization_summary = joblib.load("outputs/optimization/perfect_forecast_cvxpy_summary.pkl")
    perfect_optimization_daily = pd.read_parquet("outputs/optimization/perfect_forecast_cvxpy_daily_metrics.parquet")

    day_optimized_dispatch = pd.read_parquet("outputs/optimization/optimized_dispatch_2026_24.parquet")
    day_optimization_summary = joblib.load("outputs/optimization/24_forecast_cvxpy_summary.pkl")
    day_optimization_daily = pd.read_parquet("outputs/optimization/24_forecast_cvxpy_daily_metrics.parquet")

print("\nFORECAST-BASED CVXPY OPTIMIZATION (24-HOUR HORIZON)")
print("---------------------------------")
for metric, value in day_optimization_summary.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

print("\nFORECAST-BASED CVXPY OPTIMIZATION (4-HOUR HORIZON)")
print("---------------------------------")
for metric, value in optimization_summary.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

print("\nFORECAST-BASED CVXPY OPTIMIZATION (PERFECT)")
print("---------------------------------")
for metric, value in perfect_optimization_summary.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

"""
RULES-BASED STRATEGY
"""
if USE_RULES_BACKTEST_CACHE == False:
    rules_dispatch = rules_based_backtest(backtest_predictions)
    rules_summary, rules_daily = calculate_optimization_metrics(rules_dispatch)

    rules_dispatch.to_parquet("outputs/optimization/rules/rules_dispatch_2026.parquet", index = False)
    joblib.dump(rules_summary, "outputs/optimization/rules/forecast_rules_summary.pkl")
    rules_daily.to_parquet("outputs/optimization/rules/forecast_rules_daily_metrics.parquet", index = False)
    with open(
        "outputs/optimization/rules/forecast_rules_summary.json",
        "w"
    ) as f:
        json.dump(rules_summary, f, indent = 4)
else:
    rules_dispatch = pd.read_parquet("outputs/optimization/rules/rules_dispatch_2026.parquet")
    rules_summary = joblib.load("outputs/optimization/rules/forecast_rules_summary.pkl")
    rules_daily = pd.read_parquet("outputs/optimization/rules/forecast_rules_daily_metrics.parquet")

print("\nRULES-BASED STRATEGY")
print("--------------------")
for key, value in rules_summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:,.4f}")
    else:
        print(f"{key}: {value}")
print("\nAction counts:")
print(rules_dispatch["Action"].value_counts())
print("\nSimultaneous intervals:", ((rules_dispatch["Charge MW"] > 1e-6) & (rules_dispatch["Discharge MW"] > 1e-6)).sum())
assert rules_dispatch["SOC MWh"].min() >= 10.0 - 1e-6
assert rules_dispatch["SOC MWh"].max() <= 100.0 + 1e-6

"""
REINFORCEMENT LEARNING
"""
if USE_RL_CACHE == False:

    # Reduce threads to prevent python crashing
    import torch
    torch.set_num_threads(1)
    #from stable_baselines3 import PPO
    #model = PPO("MlpPolicy", "CartPole-v1", verbose = 1)
    #model.learn(2_000)

    # Generate environment and check to validate
    rl_env = BatteryTradingEnv(
        data = backtest_predictions,
        episode_length = 96 * 7,
        random_start = True,
    )

    check_env(
        rl_env,
        warn = True,
    )
    print("RL environment passed SB3 validation")

    observation, info = rl_env.reset(seed = 42)

    print("Initial observation:", observation)
    print("Initial info:", info)

    for _ in range(20):
        action = rl_env.action_space.sample()
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = rl_env.step(action)

        print(
            info["time_stamp"],
            info["Action Name"],
            f"SOC={info["SOC MWh"]:.2f}"
            f"Profit={info["Realized Revenue"]:.2f}"
        )
        if terminated or truncated:
            observation, info = rl_env.reset()

    # Verify SOC values
    assert (
        rl_env.minimum_soc_mwh - 1e-6
        <= info["SOC MWh"]
        <= rl_env.capacity_mwh + 1e-6
    )

    #

    """
    2021-2024: PPO training using fitted LightGBM predictions
    2025: PPO validation using holdout predictions
    2026: untouched final out-of-sample evaluation
    """
    print("Creating training data sequences...")
    predictions_2021_2024 = get_predictions(best_model, X_train, y_train, train_df["time_stamp"])
    predictions_2025 = best_predictions
    predictions_2026 = backtest_predictions

    rl_training_data = predictions_2021_2024.copy()
    rl_validation_data = predictions_2025.copy()
    rl_backtest_data = predictions_2026.copy()
    """
    Save in cache variables for future use if necessary
    """
    rl_training_data.to_parquet("outputs/cache_variables/rl_training_data.parquet", index = False)
    rl_validation_data.to_parquet("outputs/cache_variables/rl_validation_data.parquet", index = False)
    rl_backtest_data.to_parquet("outputs/cache_variables/rl_backtest_data.parquet", index = False)
    print("\nTraining data sequences completed.")


    # Verify time for steps to evaluate performance
    env = BatteryTradingEnv(
        data = rl_training_data,
        episode_length = 96 * 30, # 1 Month
        random_start = True,
    )
    obs, info = env.reset(seed = 42)
    start = time.perf_counter()
    steps = 10_000
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
    elapsed = time.perf_counter() - start
    print(f"{steps:,} steps took {elapsed:.2f} seconds")
    print(f"{steps / elapsed:,.0f} steps/second")


    print("\nTraining PPO agent on training & validation data")
    ppo_model = train_ppo_agent(
        training_data = rl_training_data,
        validation_data = rl_validation_data,
        total_timesteps = 500_000,
    )
    print("\nPPO agent training completed.")
    best_ppo_model = PPO.load(
        "models/ppo/best/best_model"
    )
    print("Initializing PPO evaluation...")
    ppo_dispatch = evaluate_ppo_agent(
        model = best_ppo_model,
        evaluation_data = rl_backtest_data
    )
    print("\n-------------\nEvaluation Complete!\n-------------")
    ppo_summary, ppo_daily = calculate_optimization_metrics(
        dispatch = ppo_dispatch,
        capacity_mwh = BATTERY_CAPACITY_MWH,
        interval_hours = INTERVAL_HOURS,
    )

    # save files
    ppo_dispatch.to_parquet("outputs/optimization/ppo/ppo_dispatch.parquet", index = False)
    ppo_daily.to_parquet("outputs/optimization/ppo/ppo_daily_metrics.parquet", index = False)
    joblib.dump(ppo_summary, "outputs/optimization/ppo/ppo_summary.pkl")
    with open(
        "outputs/optimization/ppo_summary.json",
        "w"
    ) as f:
        json.dump(ppo_summary, f, indent = 4)

else:
    ppo_dispatch = pd.read_parquet("outputs/optimization/ppo/ppo_dispatch.parquet")
    ppo_daily = pd.read_parquet("outputs/optimization/ppo/ppo_daily_metrics.parquet")
    ppo_summary = joblib.load("outputs/optimization/ppo/ppo_summary.pkl")

print(ppo_dispatch["Action Name"].value_counts())
print(
    "PPO total profit:",
    ppo_dispatch["Realized Revenue"].sum(),
)
print(
    "PPO minimum SOC:",
    ppo_dispatch["SOC MWh"].min(),
)
print(
    "PPO maximum SOC:",
    ppo_dispatch["SOC MWh"].max(),
)
print(ppo_dispatch.head(10))

"""
VISUALIZE DATA IN PLOTLY & KALEIDO
"""
if not USE_VISUALIZATION_CACHE:
    plot_data_split_timeframe(df)

    plot_model_comparison(results)
    plot_actual_vs_predicted(best_predictions, best_model_name, start="2025-07-01", end="2025-07-08")
    plot_residual_analysis(best_predictions, best_model_name)
    plot_feature_importance(best_model, FEATURES, best_model_name)
    plot_monthly_performance(best_predictions, best_model_name)
    plot_permutation_importance(model = best_model, X_test = X_test, y_test = y_test, model_name = best_model_name)
    plot_shap_summary(best_model, X_test, best_model_name)

    plot_battery_dispatch_day(optimized_dispatch)
    plot_cumulative_optimization_profit(optimized_dispatch)
    plot_forecast_horizon_performance_comparison(optimization_summary, day_optimization_summary, perfect_optimization_summary)
    plot_forecast_horizon_cumulative_profit(optimized_dispatch, day_optimized_dispatch, perfect_optimized_dispatch)
    plot_battery_dispatch_duration_curve(optimized_dispatch, charge_power_column = "Charge MW", discharge_power_column = "Discharge MW")

    plot_strategy_performance_comparison(rules_summary, optimization_summary, ppo_summary)
    plot_ppo_action_price_distribution(ppo_dispatch)
    plot_ppo_soc_heatmap(ppo_dispatch)
