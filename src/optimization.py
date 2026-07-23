"""
Initial Battery Assumptions:
Battery Capacity (MWh): 100
Max Charge (MW): 25
Max Discharge (MW): 25
Charge Efficiency: 95%
Discharge Efficiency: 95%
Initial SOC (MWh): 50
Interval Hours: 0.25
"""
import cvxpy as cvx
import numpy as np
import pandas as pd
from config import MAX_CHARGE_MW, INITIAL_SOC_MWH, MAX_DISCHARGE_MW, MINIMUM_SOC_MWH, BATTERY_CAPACITY_MWH, DEGRADATION_COST_PER_MWH, TRANSACTION_COST_PER_MWH, INTERVAL_HOURS, CHARGE_EFFICIENCY, DISCHARGE_EFFICIENCY

def optimize_daily_dispatch(
        forecast_prices: np.ndarray,
        capacity_mwh: float = BATTERY_CAPACITY_MWH,
        max_charge_mw: float = MAX_CHARGE_MW,
        max_discharge_mw: float = MAX_DISCHARGE_MW,
        charge_efficiency: float = CHARGE_EFFICIENCY,
        discharge_efficiency: float = DISCHARGE_EFFICIENCY,
        initial_soc_mwh: float = INITIAL_SOC_MWH,
        interval_hours: float = INTERVAL_HOURS,
        degradation_cost_per_mwh: float = DEGRADATION_COST_PER_MWH,
        transaction_cost_per_mwh: float = TRANSACTION_COST_PER_MWH,
        minimum_soc_mwh: float = MINIMUM_SOC_MWH,
        enforce_terminal_soc: float = True,
) -> pd.DataFrame:
    periods = len(forecast_prices)

    charge = cvx.Variable(periods, nonneg = True)
    discharge = cvx.Variable(periods, nonneg = True)
    soc = cvx.Variable(periods + 1)
    mode = cvx.Variable(periods, boolean = True) # Mutual exclusivity between charge/discharge

    constraints = [
        soc[0] == initial_soc_mwh,
        soc >= minimum_soc_mwh,
        soc <= capacity_mwh,
        charge <= max_charge_mw,
        discharge <= max_discharge_mw,
        charge <= max_charge_mw * mode,
        discharge <= max_discharge_mw * (1 - mode),
    ]

    for t in range(periods):
        constraints.append(
            soc[t + 1]
            == soc[t]
            + charge[t] * charge_efficiency * interval_hours
            - discharge[t] / discharge_efficiency * interval_hours
        )
    
    if enforce_terminal_soc:
        constraints.append(soc[-1] == initial_soc_mwh)

    forecast_revenue = cvx.sum(cvx.multiply(forecast_prices, (discharge - charge) * interval_hours))
    throughput_mwh = cvx.sum((charge + discharge) * interval_hours)
    degradation_cost = degradation_cost_per_mwh * throughput_mwh
    transaction_cost = transaction_cost_per_mwh * throughput_mwh
    net_forecast_profit = forecast_revenue - degradation_cost - transaction_cost


    problem = cvx.Problem(cvx.Maximize(net_forecast_profit), constraints)
    problem.solve(solver = cvx.HIGHS, verbose = False) # HiGHS used for mixed-integer linear program with binary charge/discharge

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Optimization failed: {problem.status}")
    
    optimized_df_day = pd.DataFrame({
        "Forecast Price": forecast_prices,
        "Charge MW": charge.value,
        "Discharge MW": discharge.value,
        "SOC MWh": soc.value[1:]
    })
    optimized_df_day["Degradation Cost"] = (
        degradation_cost_per_mwh * (optimized_df_day["Charge MW"] + optimized_df_day["Discharge MW"]) * interval_hours
    )
    optimized_df_day["Transaction Cost"] = (
        transaction_cost_per_mwh * (optimized_df_day["Charge MW"] + optimized_df_day["Discharge MW"]) * interval_hours
    )
    return optimized_df_day

def optimize_backtest_rolling(
        predictions: pd.DataFrame,
        horizon_intervals: int = 16, # 16 intervals * 0.25 hours = 4 hours of foresight
) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["time_stamp"] = pd.to_datetime(predictions["time_stamp"])
    predictions = predictions.sort_values("time_stamp").reset_index(drop = True)

    prices = predictions["Predicted"].values
    n = len(prices)
    soc = INITIAL_SOC_MWH
    all_intervals = []

    for t in range(n):
        window = prices[t : t + horizon_intervals]
        dispatch = optimize_daily_dispatch(
            forecast_prices = window,
            initial_soc_mwh = soc,
            enforce_terminal_soc = False, # 4-hour window cannot return to a fixed SOC without strangling dispatch
        )
        first = dispatch.iloc[0] # Commit only the first interval, then re-solve with a fresh window
        soc = first["SOC MWh"]

        all_intervals.append({
            "time_stamp": predictions["time_stamp"].iloc[t],
            "Forecast Price": first["Forecast Price"],
            "Charge MW": first["Charge MW"],
            "Discharge MW": first["Discharge MW"],
            "SOC MWh": soc,
            "Degradation Cost": first["Degradation Cost"],
            "Transaction Cost": first["Transaction Cost"],
        })

    result = pd.DataFrame(all_intervals)
    result = calculate_realized_profit(result, predictions["Actual"].values)

    result["Cumulative Profit"] = (
        result["Realized Revenue"].cumsum()
    )
    return result

def optimize_backtest_perfect_foresight(predictions: pd.DataFrame) -> pd.DataFrame:
    perfect = predictions.copy()
    perfect["Predicted"] = perfect["Actual"] # Dispatch against actual prices = perfect foresight
    return optimize_backtest(perfect)

def calculate_realized_profit(
        dispatch: pd.DataFrame,
        actual_prices: np.ndarray,
        interval_hours: float = 0.25,
) -> pd.DataFrame:
    result = dispatch.copy()
    result["Actual Price"] = actual_prices
    result["Gross Realized Revenue"] = (result["Discharge MW"] - result["Charge MW"]) * interval_hours * result["Actual Price"]
    result["Realized Revenue"] = result["Gross Realized Revenue"] - result["Degradation Cost"] - result["Transaction Cost"]

    result["Cumulative Profit"] = result["Realized Revenue"].cumsum()
    return result

def optimize_backtest(predictions: pd.DataFrame):
    predictions = predictions.copy()
    predictions["time_stamp"] = pd.to_datetime(predictions["time_stamp"])
    predictions = predictions.sort_values("time_stamp").reset_index(drop = True)
    all_days = []

    for date, day in predictions.groupby(predictions["time_stamp"].dt.date):
        day = day.sort_values("time_stamp").reset_index(drop = True)
        dispatch = optimize_daily_dispatch(forecast_prices = day["Predicted"].values)
        dispatch["time_stamp"] = day["time_stamp"].values
        dispatch = calculate_realized_profit(dispatch, day["Actual"].values)

        all_days.append(dispatch)

    result = pd.concat(all_days, ignore_index = True)
    result["Cumulative Profit"] = (
        result["Realized Revenue"].cumsum()
    )
    return result

def calculate_optimization_metrics(dispatch: pd.DataFrame, capacity_mwh: float = 100.0, interval_hours: float = 0.25) -> tuple[dict, pd.DataFrame]:
    required_columns = {
        "time_stamp",
        "Charge MW",
        "Discharge MW",
        "SOC MWh",
        "Realized Revenue",
        "Cumulative Profit",
    }

    results = dispatch.copy()
    results["time_stamp"] = pd.to_datetime(results["time_stamp"])
    results = results.sort_values("time_stamp").reset_index(drop = True)

    results["Charged Energy MWh"] = (results["Charge MW"] * interval_hours)
    results["Discharged Energy MWh"] = (results["Discharge MW"] * interval_hours) 
    results["Energy Throughput MWh"] = results["Charged Energy MWh"] + results["Discharged Energy MWh"]
    results["date"] = results["time_stamp"].dt.date

    daily = (
        results.groupby("date", as_index = False)
        .agg(
            Daily_Profit = ("Realized Revenue", "sum"),
            Charged_MWh = ("Charged Energy MWh", "sum"),
            Discharged_MWh = ("Discharged Energy MWh", "sum"),
            Throughput_MWh = ("Energy Throughput MWh", "sum"),
            Average_SOC_MWh = ("SOC MWh", "mean"),
            Minimum_SOC_MWh = ("SOC MWh", "min"),
            Maximum_SOC_MWh = ("SOC MWh", "max"),
            Charge_Intervals = ("Charge MW", lambda x: int((x > 1e-6).sum())),
        )
    )

    daily["Equivalent_Full_Cycles"] = daily["Discharged_MWh"] / capacity_mwh
    daily["Cumulative_Profit"] = daily["Daily_Profit"].cumsum()
    daily["Running_Peak"] = daily["Cumulative_Profit"].cummax()
    daily["Drawdown"] = daily["Cumulative_Profit"] - daily["Running_Peak"]

    results["SOC MWh"] = results["SOC MWh"].clip(lower = 0) # clip to prevent -0.00... value

    total_profit = float(results["Realized Revenue"].sum())
    total_charged = float(results["Charged Energy MWh"].sum())
    total_discharged = float(results["Discharged Energy MWh"].sum())
    total_throughput = float(results["Energy Throughput MWh"].sum())

    daily_std = float(daily["Daily_Profit"].std(ddof = 1))
    average_daily_profit = float(daily["Daily_Profit"].mean())

    if daily_std > 0:
        daily_sharpe = float(average_daily_profit / daily_std * np.sqrt(365))
    else:
        daily_sharpe = np.nan
    
    gross_profit = results.loc[results["Realized Revenue"] > 0, "Realized Revenue"].sum()
    gross_loss = -1 * results.loc[results["Realized Revenue"] < 0, "Realized Revenue"].sum()
    profit_factor = gross_profit / gross_loss

    equity = results["Cumulative Profit"]
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = abs(drawdown.min())

    active_intervals = int(((results["Charge MW"] > 1e-6) | (results["Discharge MW"] > 1e-6)).sum())
    simultaneous_intervals = int(((results["Charge MW"] > 1e-6) & (results["Discharge MW"] > 1e-6)).sum())
    total_intervals = len(results)
    summary = {
        "backtest_start": results["time_stamp"].min().isoformat(),
        "backtest_end": results["time_stamp"].max().isoformat(),
        "backtest_days": int(len(daily)),
        "intervals": int(total_intervals),

        "total_profit_usd": total_profit,
        "average_daily_profit_usd": average_daily_profit,
        "median_daily_profit_usd": float(daily["Daily_Profit"].median()),
        "best_day_profit_usd": float(daily["Daily_Profit"].max()),
        "worst_day_profit_usd": float(daily["Daily_Profit"].min()),

        "annualized_daily_sharpe": (None if np.isnan(daily_sharpe) else daily_sharpe),
        "maximum_drawdown_usd": max_drawdown,
        "profit_factor": (None if np.isnan(profit_factor) else float(profit_factor)),
        "profitable_day_percentage": float((daily["Daily_Profit"] > 0).mean() * 100),

        "total_charged_mwh": total_charged,
        "total_discharged_mwh": total_discharged,
        "total_throughput_mwh": total_throughput,
        "equivalent_full_cycles": float(total_discharged / capacity_mwh),
        "average_cycles_per_day": float(daily["Equivalent_Full_Cycles"].mean()),

        "average_soc_mwh": float(results["SOC MWh"].mean()),
        "minimum_soc_mwh": float(results["SOC MWh"].min()),
        "maximum_soc_mwh": float(results["SOC MWh"].max()),

        "active_interval_percentage": float(active_intervals / total_intervals * 100),
        "charge_interval_percentage": float((results["Charge MW"] > 1e-6).mean() * 100),
        "discharge_interval_percentage": float((results["Discharge MW"] > 1e-6).mean() * 100),
        "simultaneous_charge_discharge_intervals": simultaneous_intervals
    }
    return summary, daily