import numpy as np
import pandas as pd
from config import (
    INITIAL_SOC_MWH,
    BATTERY_CAPACITY_MWH,
    MINIMUM_SOC_MWH,
    MAX_CHARGE_MW,
    MAX_DISCHARGE_MW,
    CHARGE_EFFICIENCY,
    DISCHARGE_EFFICIENCY,
    INTERVAL_HOURS,
    DEGRADATION_COST_PER_MWH,
    TRANSACTION_COST_PER_MWH,
    THRESHOLD_WINDOW,
    LOWER_QUANTILE,
    UPPER_QUANTILE,
    MINIMUM_HISTORY,
)

def rules_based_backtest(
    predictions: pd.DataFrame,
    initial_soc_mwh: float = INITIAL_SOC_MWH,
    capacity_mwh: float = BATTERY_CAPACITY_MWH,
    minimum_soc_mwh: float = MINIMUM_SOC_MWH,
    max_charge_mw: float = MAX_CHARGE_MW,
    max_discharge_mw: float = MAX_DISCHARGE_MW,
    charge_efficiency: float = CHARGE_EFFICIENCY,
    discharge_efficiency: float = DISCHARGE_EFFICIENCY,
    interval_hours: float = INTERVAL_HOURS,
    degradation_cost_per_mwh: float = DEGRADATION_COST_PER_MWH,
    transaction_cost_per_mwh: float = TRANSACTION_COST_PER_MWH,
    threshold_window: int = THRESHOLD_WINDOW,
    lower_quantile: float = LOWER_QUANTILE,
    upper_quantile: float = UPPER_QUANTILE,
    minimum_history: int = MINIMUM_HISTORY,
) -> pd.DataFrame:
    data = predictions.copy()
    data["time_stamp"] = pd.to_datetime(data["time_stamp"])
    data = data.sort_values("time_stamp").reset_index(drop = True)

    historical_forecasts = data["Predicted"].shift(1)

    data["Lower Threshold"] = historical_forecasts.rolling(window = threshold_window, min_periods = minimum_history).quantile(lower_quantile)
    data["Upper Threshold"] = historical_forecasts.rolling(window = threshold_window, min_periods = minimum_history).quantile(upper_quantile)

    current_soc = initial_soc_mwh
    rows = []

    for _, row in data.iterrows():
        forecast_price = float(row["Predicted"])
        actual_price = float(row["Actual"])

        lower_threshold = row["Lower Threshold"]
        upper_threshold = row["Upper Threshold"]

        charge_mw = 0.0
        discharge_mw = 0.0
        action = "Hold"

        thresholds_available = pd.notna(lower_threshold) and pd.notna(upper_threshold)
        if thresholds_available:
            lower_threshold = float(lower_threshold)
            upper_threshold = float(upper_threshold)
            if forecast_price <= lower_threshold:
                remaining_capacity_mwh = capacity_mwh - current_soc

                feasible_charge_mw = remaining_capacity_mwh / (charge_efficiency * interval_hours)

                charge_mw = max(0.0, min(max_charge_mw, feasible_charge_mw))

                if charge_mw > 1e-9:
                    action = "Charge"
            elif forecast_price >= upper_threshold:
                available_energy_mwh = current_soc - minimum_soc_mwh

                feasible_discharge_mw = available_energy_mwh * discharge_efficiency / interval_hours

                discharge_mw = max(0.0, min(max_discharge_mw, feasible_discharge_mw))

                if discharge_mw > 1e-9:
                    action = "Discharge"
        
        next_soc = (
            current_soc
            + charge_mw
            * charge_efficiency
            * interval_hours
            - discharge_mw
            / discharge_efficiency
            * interval_hours
        )
        next_soc = float(np.clip(next_soc, minimum_soc_mwh, capacity_mwh))
        gross_revenue = (discharge_mw - charge_mw) * interval_hours * actual_price
        throughput_mwh = (charge_mw + discharge_mw) * interval_hours
        degradation_cost = degradation_cost_per_mwh * throughput_mwh
        transaction_cost = transaction_cost_per_mwh * throughput_mwh
        realized_revenue = gross_revenue - degradation_cost - transaction_cost
        rows.append(
            {
                "time_stamp": row["time_stamp"],
                "Forecast Price": forecast_price,
                "Actual Price": actual_price,
                "Lower Threshold": float(lower_threshold) if thresholds_available else np.nan,
                "Upper Threshold": float(upper_threshold) if thresholds_available else np.nan,
                "Action": action,
                "Charge MW": charge_mw,
                "Discharge MW": discharge_mw,
                "SOC Start MWh": current_soc,
                "SOC MWh": next_soc,
                "Gross Realized Revenue": gross_revenue,
                "Degradation Cost": degradation_cost,
                "Transaction Cost": transaction_cost,
                "Realized Revenue": realized_revenue,
            }
        )

        current_soc = next_soc

    result = pd.DataFrame(rows)
    result["Cumulative Profit"] = result["Realized Revenue"].cumsum()

    return result