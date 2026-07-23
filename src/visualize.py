# Plots
from pathlib import Path
from typing import Any, Mapping, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance
import shap
from config import MINIMUM_SOC_MWH, BATTERY_CAPACITY_MWH

FEATURE_LABELS = {
        "lag_1": "Price 15 Minutes Ago",
        "lag_4": "Price 1 Hour Ago",
        "lag_12": "Price 3 Hours Ago",
        "lag_24": "Price 6 Hours Ago",
        "lag_96": "Price 1 Day Ago",
        "lag_672": "Price 1 Week Ago",

        "is_weekend": "Weekend Indicator",

        "rolling_mean_4": "1-Hour Rolling Mean",
        "rolling_std_4": "1-Hour Rolling Volatility",
        "rolling_mean_12": "3-Hour Rolling Mean",
        "rolling_std_12": "3-Hour Rolling Volatility",
        "rolling_mean_24": "6-Hour Rolling Mean",
        "rolling_std_24": "6-Hour Rolling Volatility",
        "rolling_mean_48": "12-Hour Rolling Mean",
        "rolling_std_48": "12-Hour Rolling Volatility",
        "rolling_mean_96": "1-Day Rolling Mean",
        "rolling_std_96": "1-Day Rolling Volatility",
        "rolling_mean_192": "2-Day Rolling Mean",
        "rolling_std_192": "2-Day Rolling Volatility",
        "rolling_mean_672": "1-Week Rolling Mean",
        "rolling_std_672": "1-Week Rolling Volatility",

        "hour": "Hour of Day",
        "day_of_week": "Day of Week",
        "month": "Month",

        "hour_sin": "Hour of Day Cycle (Sine)",
        "hour_cos": "Hour of Day Cycle (Cosine)",

        "day_of_week_sin": "Weekday Cycle (Sine)",
        "day_of_week_cos": "Weekday Cycle (Cosine)",

        "month_sin": "Annual Seasonality (Sine)",
        "month_cos": "Annual Seasonality (Cosine)",

#        "load_lag_1": "Load 15 Minutes Ago",
#        "load_lag_4": "Load 1 Hour Ago",

        "load_change_1": "15-Minute Load Change",
        "load_change_4": "1-Hour Load Change",

#        "load_pct_change_1": "15-Minute Load % Change",
#        "load_pct_change_4": "1-Hour Load % Change",
        
        "load_deviation_96": "Load Deviation from 24-Hour Average",
        "load_volatility_24": "6-Hour Load Volatility",

        "temp_lag_1": "Temperature 15 Minutes Ago",
        "temp_lag_4": "Temperature 1 Hour Ago",
        "temp_change_1": "15-Minute Temperature Change",
        "temp_change_4": "1-Hour Temperature Change",
        "temp_rolling_mean_24": "1-Day Temperature Rolling Mean",
        "temp_volatility_24": "1-Day Temperature Rolling Volatility",

        "wind_lag_1": "Wind 15 Minutes Ago",
        "wind_change_4": "1-Hour Wind Change",
        "wind_rolling_mean_24": "1-Hour Wind Rolling Mean",
        "wind_volatility_24": "1-Day Wind Rolling Volatility",

        "solar_lag_1": "Solar Radiation 15 Minutes Ago",
        "solar_change_4": "1-Hour Solar Radiation Change",
        "solar_rolling_mean_4": "1-Hour Solar Radiation Rolling Mean",
        "solar_volatility_24": "1-Day Solar Radiation Rolling Volatility",

        "cloud_lag_1": "Cloud Coverage 15 Minutes Ago",
        "cloud_change_4": "1-Hour Cloud Coverage Change",

        "dew_lag_1": "Dewpoint 15 Minutes Ago",
        "dew_change_4": "1-Hour Dewpoint Change",

        "precip_lag_1": "Precipitation 15 Minutes Ago",

        "heat_index_proxy_1": "Heat Index Proxy 15 Minutes Ago",
    }

def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents = True, exist_ok = True)

def _save_figure(fig: go.Figure, output_stem: str) -> None:
    """
    Save both an interactive HTML and high-resolution PNG with kaleido
    """
    html_path = f"{output_stem}.html"
    png_path = f"{output_stem}.png"
    _ensure_parent(html_path)
    fig.write_html(html_path)
    fig.write_image(png_path, scale = 3)

def plot_data_split_timeframe(
        df: pd.DataFrame,
        output_stem: str = "outputs/figures/data_split_timeline",
) -> None:
    plot_df = df.sort_values("time_stamp").copy()
    plot_df["time_stamp"] = pd.to_datetime(plot_df["time_stamp"])
    # switch to native python
    plot_times = plot_df["time_stamp"].dt.to_pydatetime()

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x = plot_times,
            y = plot_df["Settlement Point Price"],
            mode = "lines",
            name = "Houston Hub Price",
            line = dict(width = 1),
            hovertemplate = (
                "%{x|%b %d, %Y:%H}<br>"
                "Price: $%{y:.2f}/MWh"
                "<extra></extra>"
            ),
        )
    )
    fig.add_vrect(
        x0 = pd.Timestamp("2021-01-01").to_pydatetime(),
        x1 = pd.Timestamp("2025-01-01").to_pydatetime(),
        fillcolor = "rgba(37, 99, 235, 0.05)",
        line_width = 0,
        layer = "below",
        annotation_text = "<b>Training Data</b>",
        annotation_position = "top left",
    )
    fig.add_vrect(
        x0 = pd.Timestamp("2025-01-01").to_pydatetime(),
        x1 = pd.Timestamp("2026-01-01").to_pydatetime(),
        fillcolor = "rgba(245, 158, 11, 0.06)",
        line_width = 0,
        layer = "below",
        annotation_text = "<b>Holdout Test</b>",
        annotation_position = "top left",
    )
    fig.add_vrect(
        x0 = pd.Timestamp("2026-01-01").to_pydatetime(),
        x1 = plot_df["time_stamp"].max().to_pydatetime(),
        fillcolor = "rgba(16, 185, 129, 0.06)",
        line_width = 0,
        layer = "below",
        annotation_text = "<b>Forecast / Backtest</b>",
        annotation_position = "top left",
    )
    fig.add_vline(
        x = pd.Timestamp("2025-01-01").to_pydatetime(),
        line_dash = "dash",
        line_width = 2,
    )
    fig.add_vline(
        x = pd.Timestamp("2026-01-01").to_pydatetime(),
        line_dash = "dash",
        line_width = 2,
    )
    fig.update_layout(
        title = {
            "text": (
                "<b>ERCOT Houston Hub Price History and Modeling Split</b>"
                "<br><sup>"
                "2021-2024 training • 2025 holdout testing • "
                "2026 forecast and strategy backtest"
                "</sup>"
            ),
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis_title = "Time",
        yaxis_title = "Settlement Point Price ($/MWh)",
        template = "plotly_white",
        height = 680,
        hovermode = "x unified",
        margin = dict(l = 60, r = 50, t = 120, b = 80),
        legend = dict(orientation = "h", x = 0.5, xanchor = "center", y = 1.02, yanchor = "bottom")
    )
    fig.update_xaxes(
        rangeselector = dict(
            buttons = [
                dict(count = 6, label = "6M", step = "month", stepmode = "backward"),
                dict(count = 1, label = "1Y", step = "year", stepmode = "backward"),
                dict(count = 3, label = "3Y", step = "year", stepmode = "backward"),
                dict(step = "all", label = "All"),
            ]
        ),
        rangeslider_visible = True,
        type = "date",
    )
    _save_figure(fig, output_stem)

def plot_model_comparison(results: pd.DataFrame, output_steam: str = "outputs/figures/model_comparison") -> None:
    plot_df = results.sort_values("RMSE", ascending = True).copy()
    baseline_rmse = plot_df.loc[plot_df["Model"] == "Linear Regression", "RMSE"].iloc[0]
    plot_df["Improvement"] = (baseline_rmse - plot_df["RMSE"]) / baseline_rmse * 100
    best_model_name = plot_df.loc[plot_df["RMSE"].idxmin(), "Model"]
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x = plot_df["RMSE"],
            y = plot_df["Model"],
            orientation = "h",
            text = plot_df.apply(
                lambda row: (
                    f"{row["RMSE"]:.2f}"
                    if row["Model"] == "Linear Regression"
                    else f"<b>{row["RMSE"]:.2f}</b> ({row["Improvement"]:.1f}% better)"
                ),
                axis = 1,
            ),
            textposition = "outside",
            customdata = plot_df[["MAE", "R2", "Improvement"]].to_numpy(),
            hovertemplate = (
                "<b>%{y}</b><br>"
                "RMSE: %{x:.3f}<br>"
                "MAE: %{customdata[0]:.3f}<br>"
                "R²: %{customdata[1]:.3f}<br>"
                "RMSE improvement vs Linear: %{customdata[2]:.1f}%"
                "<extra></extra>"
            )
        )
    )
    best_row = plot_df.loc[plot_df["Model"] == best_model_name].iloc[0]
    fig.add_annotation(
        x = best_row["RMSE"],
        y = best_model_name,
        text = "Best",
        showarrow = True,
        arrowhead = 2,
        ax = 15,
        ay = -28,
        font = dict(size = 12)
    )

    fig.update_layout(
        title = {
            "text": (
                "<b>Forecast Model Performance</b>"
                "<br><sup>"
                "2025 holdout period • Lower RMSE indicates better performance"
                "</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title = "RMSE ($/MWh)",
        yaxis_title = "",
        template = "plotly_white",
        height = 540,
        margin = dict(l = 175, r = 180, t = 115, b = 65),
        showlegend = False
    )
    fig.update_traces(cliponaxis = False)

    fig.update_xaxes(range = [0, plot_df["RMSE"].max() * 1.32])
    fig.update_yaxes(autorange = "reversed")
    _save_figure(fig, output_steam)

def plot_actual_vs_predicted(predictions: pd.DataFrame, model_name: str, start: str, end: str, output_stem: str = "outputs/figures/actual_vs_predicted") -> None:
    plot_df = predictions.copy()
    if start is not None:
        plot_df = plot_df[plot_df["time_stamp"] >= start]
    if end is not None:
        plot_df = plot_df[plot_df["time_stamp"] <= end]
    
    # mae, rmse, r2 for the sample showcased for accuracy
    mae = plot_df["Absolute Error"].mean()
    rmse = (plot_df["Error"].pow(2).mean()) ** 0.5
    r2 = 1 - ((plot_df["Error"] ** 2).sum() / ((plot_df["Actual"] - plot_df["Actual"].mean()) ** 2).sum())
    start_label = plot_df["time_stamp"].min().strftime("%b, %d, %Y")
    end_label = plot_df["time_stamp"].max().strftime("%b, %d, %Y")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = plot_df["time_stamp"],
            y = plot_df["Actual"],
            name = "Actual",
            mode = "lines",
            line = dict(width = 2.8),
            hovertemplate = (
                "%{x|%b %d, %Y %H:%M}<br>"
                "Actual: $%{y:.2f}/MWh"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x = plot_df["time_stamp"],
            y = plot_df["Predicted"],
            name = "Predicted",
            mode = "lines",
            line = dict(width = 2.2, dash = "dot"),
            hovertemplate = (
                "%{x|%b %d, %Y %H:%M}<br>"
                "Predicted: $%{y:.2f}/MWh"
                "<extra></extra>"
            )
        )
    )
    
    peak_idx = plot_df["Actual"].idxmax()
    peak_time = plot_df.loc[peak_idx, "time_stamp"].to_pydatetime()
    peak_price = plot_df.loc[peak_idx, "Actual"]
    peak_prediction = plot_df.loc[peak_idx, "Predicted"]
    absolute_peak_error = abs(peak_price - peak_prediction)

    fig.add_annotation(
        x = peak_time,
        y = peak_price,
        text = (
            "<b>Peak Price Event</b><br>"
            f"Actual: ${peak_price:.2f}/MWh<br>"
            f"Forecast: ${peak_prediction:.2f}/MWh<br>"
            f"Absolute Error: ${absolute_peak_error:.2f}/MWh"
        ),
        showarrow = True,
        arrowhead = 2,
        arrowsize = 1,
        arrowwidth = 1.5,
        ax = 75,
        ay = -75,
        bgcolor = "rgba(255,255,255,0.9)",
        bordercolor = "gray",
        borderwidth = 1,
        borderpad = 6,
        align = "left",
    )
    fig.add_trace(
        go.Scatter(
            x = [peak_time],
            y = [peak_price],
            mode = "markers",
            name = "Peak price",
            marker = dict(size = 10, symbol = "diamond"),
            hovertemplate = (
                "<b>Peak Price Event</b><br>"
                "%{x|%b %d, %Y %H:%M}<br>"
                f"Actual: ${peak_price:.2f}/MWh<br>"
                f"Forecast: ${peak_prediction:.2f}/MWh"
                "<extra></extra>"
            ),
            showlegend = False,
        )
    )

    fig.update_layout(
        title = {
            "text": (
                f"<b>{model_name}: Actual vs Predicted ERCOT Prices</b>"
                "<br><sup>"
                f"2025 holdout sample • {start_label}-{end_label} • "
                f"MAE = <b>{mae:.2f}</b> • RMSE = <b>{rmse:.2f}</b> • R² = <b>{r2:.3f}</b>"
                "</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title = "Time",
        yaxis_title = "Settlement Point Price ($/MWh)",
        template = "plotly_white",
        height = 650,
        hovermode = "x unified",
        legend = dict(
            orientation = "h",
            yanchor = "bottom",
            y = 1.02,
            xanchor = "center",
            x = 0.5
        ),
        margin = dict(l = 80, r = 45, t = 120, b = 70),
    )
    fig.update_xaxes(rangeslider_visible = True, rangeselector = dict(
        buttons = [
            dict(count = 12, label = "12H", step = "hour", stepmode = "backward"),
            dict(count = 1, label = "1D", step = "day", stepmode = "backward"),
            dict(count = 3, label = "3D", step = "day", stepmode = "backward"),
            dict(step = "all", label = "All"),
        ]
    ))
    _save_figure(fig, output_stem)

def plot_residual_analysis(predictions: pd.DataFrame, model_name: str, output_stem: str = "outputs/figures/residual_analysis") -> None:
    fig = make_subplots(
        rows = 2,
        cols = 2,
        subplot_titles = (
            "Residuals Through Time",
            "Residual Distribution",
            "Actual Price vs Residual",
            "Absolute Error Through Time"
        ),
        vertical_spacing = 0.14,
        horizontal_spacing = 0.1
    )

    plot_df = predictions.copy()
    residual_low = plot_df["Error"].quantile(0.005)
    residual_high = plot_df["Error"].quantile(0.995)
    histogram_errors = plot_df.loc[plot_df["Error"].between(residual_low, residual_high), "Error"]

    fig.add_trace(
        go.Scatter(
            x = predictions["time_stamp"],
            y = predictions["Error"],
            mode = "markers",
            marker = dict(size = 3, opacity = 0.45),
            name = "Residual"
        ),
        row = 1,
        col = 1
    )

    fig.add_trace(
        go.Histogram(
            x = histogram_errors,
            nbinsx = 120,
            name = "Residual distribution",
            hovertemplate = (
                "Residual range: %{x}<br>"
                "Observations: %{y}"
                "<extra></extra>"
            ),
        ),
        row = 1,
        col = 2
    )

    fig.add_annotation(
        text = "Histogram displays the central 99% of residuals",
        xref = "x2 domain",
        yref = "y2 domain",
        x = 0.5,
        y = 1.02,
        showarrow = False,
        font = dict(size = 11)
    )

    fig.add_trace(
        go.Scatter(
            x = predictions["Actual"],
            y = predictions["Error"],
            mode = "markers",
            marker = dict(size = 4, opacity = 0.35),
            name = "Actual vs residual"
        ),
        row = 2,
        col = 1
    )

    fig.add_trace(
        go.Scatter(
            x = predictions["time_stamp"],
            y = predictions["Absolute Error"],
            mode = "lines",
            line = dict(width = 1),
            name = "Absolute error"
        ),
        row = 2,
        col = 2
    )

    fig.add_hline(y = 0, line_dash = "dash", row = 1, col = 1)
    fig.add_hline(y = 0, line_dash = "dash", row = 2, col = 1)

    fig.update_layout(
        title = {
            "text": (
                f"<b>{model_name}: Forecast Error Diagnostics</b>"
                "<br><sup>"
                "2025 holdout period • Residual = actual price - predicted price"
                "</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97
        },
        template = "plotly_white",
        height = 850,
        showlegend = False,
        margin = dict(l = 60, r = 40, t = 160, b = 60)
    )

    fig.update_xaxes(title_text = "Time", row = 1, col = 1)
    fig.update_yaxes(title_text = "Residual ($/MWh)", row = 1, col = 1)

    fig.update_xaxes(title_text = "Residual ($/MWh)", row = 1, col = 2)
    fig.update_yaxes(title_text = "Count", row = 1, col = 2)

    fig.update_xaxes(title_text = "Actual Price ($/MWh)", row = 2, col = 1)
    fig.update_yaxes(title_text = "Residual ($/MWh)", row = 2, col = 1)

    fig.update_xaxes(title_text = "Time", row = 2, col = 2)
    fig.update_yaxes(title_text = "Absolute Error ($/MWh)", row = 2, col = 2)
    _save_figure(fig, output_stem)

def plot_feature_importance(model, features: list[str], model_name: str, top_n: int = 20, output_stem: str = "outputs/figures/feature_importance") -> pd.DataFrame:
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df.sort_values("Importance", ascending = False).head(top_n).sort_values("Importance", ascending = True)
    )

    importance_df["Feature"] = (
        importance_df["Feature"].map(FEATURE_LABELS).fillna(importance_df["Feature"]) # Convert feature labels
    )
    importance_df["Importance"] = (importance_df["Importance"] / importance_df["Importance"].sum() * 100)


    fig = px.bar(
        importance_df,
        x = "Importance",
        y = "Feature",
        orientation = "h",
        text = "Importance",
    )
    fig.update_traces(
        texttemplate = "%{text:.1f}%",
        textposition = "outside",
        hovertemplate = "<b>%{y}</b><br>Share of total importance: %{x:.2f}<extra></extra>"
    )

    fig.update_layout(
        title = {
            "text": (
                f"{model_name}: <b>Top {top_n} Forecast Drivers</b>"
                "<br><sup>"
                f"{model_name} split importance • 2021-2024 training period"
                "</sup>"
            ),
        },
        template = "plotly_white",
        height = 720,
        xaxis_title = "Feature Importance",
        yaxis_title = " ",
        margin = dict(l = 170, r = 60, t = 90, b = 60)
    )
    _save_figure(fig, output_stem)
    return importance_df

def plot_monthly_performance(predictions: pd.DataFrame, best_model_name: str, output_stem: str = "outputs/figures/monthly_performance") -> pd.DataFrame:
    plot_df = predictions.copy()
    plot_df["Month"] = plot_df["time_stamp"].dt.to_period("M").astype(str)
    plot_df["Squared Error"] = plot_df["Error"] ** 2

    monthly = (
        plot_df.groupby("Month", as_index = False).agg(
            MAE = ("Absolute Error", "mean"),
            MSE = ("Squared Error", "mean")
        )
    )
    monthly["RMSE"] = monthly["MSE"] ** 0.5
    monthly["Month"] = pd.to_datetime(monthly["Month"]).dt.strftime("%b")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = monthly["Month"],
            y = monthly["MAE"],
            name = "MAE",
            mode = "lines+markers",
            line = dict(width = 2.5),
            marker = dict(size = 8),
            hovertemplate = "<b>%{x}</b><br>MAE: $%{y:.2f}/MWh<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x = monthly["Month"],
            y = monthly["RMSE"],
            name = "RMSE",
            mode = "lines+markers+text",
            line = dict(width = 2.5),
            marker = dict(size = 8),
            hovertemplate = "<b>%{x}</b><br>RMSE: $%{y:.2f}/MWh<extra></extra>"
        )
    )

    fig.update_layout(
        title = {
            "text": (
                "<b>Monthly Forecast Performance</b>"
                "<br><sup>"
                f"{best_model_name} evaluated on the 2025 holdout period"
                "</sup>"
            ),
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis_title = "Month",
        yaxis_title = "Forecast Error ($/MWh)",
        template = "plotly_white",
        height = 600,
        barmode = "group",
        hovermode = "x unified",
        legend = dict(orientation = "h", yanchor = "bottom", y = 1.02, xanchor = "center", x = 0.5),
        margin = dict(l = 75, r = 45, t = 115, b = 70)
    )

    fig.update_traces(
        selector = dict(name = "RMSE"),
        text = monthly["RMSE"].round(2),
        textposition = "top center",
    )
    _save_figure(fig, output_stem)
    return monthly

def plot_permutation_importance(model, X_test: pd.DataFrame, y_test: pd.DataFrame, model_name: str, output_stem: str = "outputs/figures/permutation_importance", top_n: int = 20, n_repeats: int = 5) -> pd.DataFrame:
    result = permutation_importance(estimator = model, X = X_test, y = y_test, scoring = "neg_root_mean_squared_error", n_repeats = n_repeats, random_state = 42, n_jobs = 1)
    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean,
        "Importance_std": result.importances_std,
    })

    importance_df = (
        importance_df.sort_values("Importance", ascending = False).head(top_n).sort_values("Importance", ascending = True)
    )

    importance_df["Feature_Label"] = (
        importance_df["Feature"].map(FEATURE_LABELS).fillna(importance_df["Feature"]) # Convert feature labels
    )

    plot_df = importance_df.head(top_n).sort_values("Importance", ascending = True)

    fig = go.Figure(
        go.Bar(
            x = plot_df["Importance"],
            y = plot_df["Feature_Label"],
            orientation = "h",
            error_x = {
                "type": "data",
                "array": plot_df["Importance_std"],
                "visible": True,
                "thickness": 1,
                "width": 3,
            },
            text = plot_df["Importance"].map(lambda value: f"{value:.2f}"),
            textposition = "outside",
            customdata = np.column_stack([plot_df["Importance_std"], plot_df["Feature"]]),
            hovertemplate = (
                "<b>%{y}</b><br>"
                "RMSE increase: %{x:.3f}<br>"
                "Standard deviation: %{customdata[0]:.3f}<br>"
                "Variable: %{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title = {
            "text": (
                f"<b>{model_name}: Permutation Importance</b>"
                "<br><sup>"
                "Out-of-sample importance • 2025 holdout test period"
                "</sup>"
            ),
            "x": 0.04,
            "xanchor": "left",
        },
        xaxis_title = "Increase in Holdout RMSE ($/MWh)",
        yaxis_title = "",
        template = "plotly_white",
        height = max(650, top_n * 34),
        margin = {"l": 280, "r": 90, "t": 100, "b": 80},
        showlegend = False,
        hoverlabel = {"font_size": 13},
    )
    fig.update_xaxes(
        showgrid = True,
        gridcolor = "rgba(0, 0, 0, 0.08)",
        zeroline = True,
        zerolinecolor = "rgba(0, 0, 0, 0.3)",
    )
    fig.update_yaxes(
        showgrid = False,
        automargin = True,
    )
    _save_figure(fig, output_stem)
    return importance_df

def plot_shap_summary(model, X_test: pd.DataFrame, model_name: str, output_stem: str = "outputs/figures/shap_summary", top_n: int = 20, sample_size: int = 5000, random_state: int = 42) -> pd.DataFrame:
    X_sample = X_test.sample(n = min(sample_size, len(X_test)), random_state = random_state).copy() # reduce sample size
    
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)
    shap_values = np.asarray(explanation.values)

    mean_abs_shap = np.abs(shap_values).mean(axis = 0)
    importance_df = pd.DataFrame(
        {
            "Feature": X_sample.columns,
            "Mean Absolute SHAP": mean_abs_shap,
        }
    ).sort_values("Mean Absolute SHAP", ascending = False)

    top_features = importance_df.head(top_n)["Feature"].tolist()
    plot_features = list(reversed(top_features))

    rng = np.random.default_rng(random_state)
    fig = go.Figure()
    for row_position, feature in enumerate(plot_features):
        column_position = X_sample.columns.get_loc(feature)
        feature_shap = shap_values[:, column_position]
        feature_values = pd.to_numeric(X_sample[feature], errors = "coerce").to_numpy(dtype = float)

        finite_mask = np.isfinite(feature_values)
        finite_values = feature_values[finite_mask]
        if len(finite_values) == 0:
            normalized_values = np.full(len(feature_values), 0.5)
        else:
            lower = np.nanpercentile(finite_values, 5)
            upper = np.nanpercentile(finite_values, 95)

            if np.isclose(lower, upper):
                normalized_values = np.full(len(feature_values), 0.5)
            else:
                normalized_values = (np.clip(feature_values, lower, upper) - lower) / (upper - lower)

                normalized_values = np.nan_to_num(normalized_values, nan = 0.5, posinf = 1.0, neginf = 0.0)
        jitter = rng.normal(loc = 0.0, scale = 0.11, size = len(feature_shap))

        y_values = row_position + jitter
        feature_label = FEATURE_LABELS.get(feature, feature)

        fig.add_trace(
            go.Scattergl(
                x = feature_shap,
                y = y_values,
                mode = "markers",
                name = feature_label,
                showlegend = False,
                marker = {
                    "size": 5,
                    "opacity": 0.58,
                    "color": normalized_values,
                    "colorscale": [
                        [0.00, "#2563EB"],
                        [0.50, "#D9D9E3"],
                        [1.00, "#E11D48"]
                    ],
                    "cmin": 0,
                    "cmax": 1,
                    "showscale": row_position == len(plot_features) - 1,
                    "colorbar": {
                        "title": {
                            "text": "Feature Value",
                            "side": "right",
                        },
                        "tickmode": "array",
                        "tickvals": [0, 1],
                        "ticktext": ["Low", "High"],
                        "thickness": 14,
                        "len": 0.58,
                        "y": 0.5,
                        "outlinewidth": 0,
                    },
                },
                customdata = np.column_stack([feature_values, np.repeat(feature_label, len(feature_values))]),
                hovertemplate = (
                    "<b>%{customdata[1]}</b><br>"
                    "Feature value: %{customdata[0]:.3f}<br>"
                    "SHAP contribution: %{x:,.3f} $/MWh"
                    "<extra></extra>"
                ),
            )
        )
    fig.add_vline(
        x = 0,
        line_width = 1.2,
        line_dash = "dash",
        line_color = "rgba(40, 40, 40, 0.55)",
    )
    fig.update_layout(
        title = {
            "text": (
                f"<b>{model_name}: SHAP Summary</b>"
                "<br>"
                "<sup>"
                f"Feature impact across {len(X_sample):,} sampled"
                "observations from the 2025 holdout period"
                "</sup>"
            ),
            "x": 0.045,
            "xanchor": "left",
            "y": 0.975,
            "yanchor": "top",
        },
        template = "plotly_white",
        height = max(720, top_n * 38),
        margin = {"l": 260, "r": 135, "t": 105, "b": 85},
        xaxis_title = "Impact on Predicted Settlement Price ($/MWh)",
        yaxis_title = "",
        hoverlabel = {"font_size": 13},
        plot_bgcolor = "white",
        paper_bgcolor = "white",
    )
    fig.update_xaxes(
        showgrid = True,
        gridcolor = "rgba(0, 0, 0, 0.08)",
        zeroline = False,
        ticks = "outside",
        title_standoff = 18,
    )
    fig.update_yaxes(
        tickmode = "array",
        tickvals = list(range(len(plot_features))),
        ticktext = [
            FEATURE_LABELS.get(feature, feature)
            for feature in plot_features
        ],
        showgrid = False,
        zeroline = False,
        automargin = True,
        fixedrange = False,
    )
    fig.add_annotation(
        x = 1,
        y = -0.105,
        xref = "paper",
        yref = "paper",
        text = (
            "<span style='color:#2563EB'>Low feature value</span>"
            " &nbsp;•&nbsp; "
            "<span style='color:#E11D48'>High feature value</span>"
        ),
        showarrow = False,
        xanchor = "right",
        font = {"size": 11},
    )
    _save_figure(fig, output_stem)
    return importance_df

def plot_battery_dispatch_day(
        dispatch: pd.DataFrame,
        date: str | None = None,
        output_stem: str = "outputs/figures/battery_dispatch_day"
) -> pd.DataFrame:
    data = dispatch.copy()
    data["time_stamp"] = pd.to_datetime(data["time_stamp"])
    data = data.sort_values("time_stamp").reset_index(drop = True)
    data["date"] = data["time_stamp"].dt.date

    if date is None:
        daily_profit = data.groupby("date")["Realized Revenue"].sum().sort_values(ascending = False)
        selected_date = daily_profit.index[0]
    else:
        selected_date = pd.Timestamp(date).date()
    
    day = data[data["date"] == selected_date].copy()

    day["Net Dispatch MW"] = day["Discharge MW"] - day["Charge MW"]
    daily_profit_usd = day["Realized Revenue"].sum()

    fig = make_subplots(
        rows = 3,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing = 0.055,
        row_heights = [0.42, 0.28, 0.30],
        specs = [
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )
    fig.add_trace(
        go.Scatter(
            x = day["time_stamp"],
            y = day["Actual Price"],
            mode = "lines",
            name = "Actual Price",
            line = {"width": 2.4, "color": "#111827"},
            hovertemplate = (
                "<b>Actual Price</b><br>"
                "%{x|%I:%M %p}<br>"
                "$%{y:,.2f}/MWh"
                "<extra></extra>"
            ),
        ),
        row = 1,
        col = 1,
    )
    fig.add_trace(
        go.Scatter(
            x = day["time_stamp"],
            y = day["Forecast Price"],
            mode = "lines",
            name = "Forecast Price",
            line = {"width": 2.0, "dash": "dot", "color": "#6366F1"},
            hovertemplate = (
                "<b>Forecast Price</b><br>"
                "%{x|%I:%M %p}<br>"
                "$%{y:,.2f}/MWh"
                "<extra></extra>"
            ),
        ),
        row = 1,
        col = 1,
    )
    # panel 2
    fig.add_trace(
        go.Bar(
            x = day["time_stamp"],
            y = day["Discharge MW"],
            name = "Discharge",
            marker_color = "#2563EB",
            opacity = 0.84,
            hovertemplate = (
                "<b>Discharge</b><br>"
                "%{x|%I:%M %p}<br>"
                "%{y:,.2f} MW"
                "<extra></extra>"
            ),
        ),
        row = 2,
        col = 1,
    )
    fig.add_trace(
        go.Bar(
            x = day["time_stamp"],
            y = -day["Charge MW"],
            name = "Charge",
            marker_color = "#94A3B8",
            opacity = 0.84,
            hovertemplate = (
                "<b>Charge</b><br>"
                "%{x|%I:%M %p}<br>"
                "%{customdata:,.2f} MW"
                "<extra></extra>"
            ),
            customdata = day["Charge MW"],
        ),
        row = 2,
        col = 1,
    )
    fig.add_trace(
        go.Scatter(
            x = day["time_stamp"],
            y = day["SOC MWh"],
            mode = "lines",
            name = "State of Charge",
            line = {"width": 2.5, "color": "#0F766E"},
            fill = "tozeroy",
            fillcolor = "rgba(15, 118, 110, 0.10)",
            hovertemplate = (
                "<b>State of Charge</b><br>"
                "%{x|%I:%M %p}<br>"
                "%{y:,.2f} MWh"
                "<extra></extra>"
            ),
        ),
        row = 3,
        col = 1,
    )
    fig.add_hline(
        y = 0,
        row = 2,
        col = 1,
        line_width = 1.2,
        line_color = "rgba(15, 23, 42, 0.55)",
    )
    fig.add_hline(
        y = MINIMUM_SOC_MWH,
        row = 3,
        col = 1,
        line_width = 1.2,
        line_dash = "dot",
        line_color = "rgba(100, 116, 139, 0.65)",
        annotation_text = f"{MINIMUM_SOC_MWH} MWh reserve",
        annotation_position = "top right",
    )
    fig.update_layout(
        title = {
            "text": (
                "<b>Forecast-Driven Battery Dispatch</b>"
                "<br>"
                "<sup>"
                f"{selected_date:%B %d, %Y} • "
                f"Realized net profit: ${daily_profit_usd:,.0f}"
                "</sup>"
            ),
            "x": 0.045,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
        template = "plotly_white",
        height = 920,
        barmode = "relative",
        hovermode = "x unified",
        margin = {"l": 85, "r": 55, "t": 105, "b": 75},
        legend = {"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
        plot_bgcolor = "white",
        paper_bgcolor = "white",
    )
    fig.update_yaxes(
        title_text = "Price ($/MWh)",
        row = 1,
        col = 1,
        gridcolor = "rgba(15, 23, 42, 0.08)",
        zeroline = False,
    )
    fig.update_yaxes(
        title_text = "Dispatch (MW)",
        row = 2,
        col = 1,
        gridcolor = "rgba(15, 23, 42, 0.08)",
        zeroline = False,
    )
    fig.update_yaxes(
        title_text = "SOC (MWh)",
        range = [0, 105],
        row = 3,
        col = 1,
        gridcolor = "rgba(15, 23, 42, 0.08)",
        zeroline = False,
    )
    fig.update_xaxes(
        showgrid = False,
        tickformat = "%I:%M %p",
        row = 3,
        col = 1,
        title_text = "Time"
    )
    _save_figure(fig, output_stem)
    return day

def plot_cumulative_optimization_profit(
    dispatch: pd.DataFrame,
    output_stem: str = "outputs/figures/cumulative_optimization_profit"
) -> pd.DataFrame:
    data = dispatch.copy()
    data["time_stamp"] = pd.to_datetime(data["time_stamp"])
    data = data.sort_values("time_stamp").reset_index(drop = True)
    data["date"] = data["time_stamp"].dt.normalize()

    daily = (
        data.groupby("date", as_index = False)
        .agg(
            Daily_Profit = ("Realized Revenue", "sum")
        )
    )
    daily["Cumulative Profit"] = daily["Daily_Profit"].cumsum()
    daily["Running Peak"] = daily["Cumulative Profit"].cummax()
    daily["Drawdown"] = daily["Cumulative Profit"] - daily["Running Peak"]

    total_profit = daily["Cumulative Profit"].iloc[-1]
    max_drawdown = abs(daily["Drawdown"].min())

    best_day_index = daily["Daily_Profit"].idxmax()
    worst_day_index = daily["Daily_Profit"].idxmin()

    best_day = daily.iloc[best_day_index]
    worst_day = daily.iloc[worst_day_index]

    fig = make_subplots(
        rows = 2,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing = 0.06,
        row_heights = [0.76, 0.24],
    )
    fig.add_trace(
        go.Scatter(
            x = daily["date"],
            y = daily["Cumulative Profit"],
            mode = "lines",
            name = "Cumulative Profit",
            line = {"width": 2.8, "color": "#2563EB"},
            fill = "tozeroy",
            fillcolor = "rgba(37, 99, 235, 0.08)",
            hovertemplate = (
                "<b>%{x|%b %d, %Y}</b><br>"
                "Cumulative profit: $%{y:,.0f}"
                "<extra></extra>"
            ),
        ),
        row = 1,
        col = 1,
    )
    fig.add_trace(
        go.Scatter(
            x = daily["date"],
            y = daily["Drawdown"],
            mode = "lines",
            name = "Drawdown",
            line = {"width": 2.0, "color": "#64748B"},
            fill = "tozeroy",
            fillcolor = "rgba(100, 116, 139, 0.13)",
            hovertemplate = (
                "<b>%{x|%b %d, %Y}</b><br>"
                "Drawdown: $%{y:,.0f}"
                "<extra></extra>"
            ),
        ),
        row = 2,
        col = 1,
    )
    fig.add_annotation(
        x = best_day["date"].to_pydatetime(),
        y = best_day["Cumulative Profit"],
        text = (
            f"Best day<br>"
            f"+${best_day["Daily_Profit"]:,.0f}"
        ),
        showarrow = True,
        arrowhead = 2,
        arrowsize = 1,
        arrowwidth = 1,
        arrowcolor = "rgba(15, 23, 42, 0.55)",
        ax = 0,
        ay = -55,
        bgcolor = "rgba(255, 255, 255, 0.92)",
        bordercolor = "rgba(15, 23, 42, 0.15)",
        borderwidth = 1,
        borderpad = 5,
        font = {"size": 11},
        row = 1,
        col = 1,
    )
    fig.add_annotation(
        x = worst_day["date"].to_pydatetime(),
        y = worst_day["Cumulative Profit"],
        text = (
            f"Worst day<br>"
            f"${worst_day["Daily_Profit"]:,.0f}"
        ),
        showarrow = True,
        arrowhead = 2,
        arrowsize = 1,
        arrowcolor = "rgba(15, 23, 42, 0.55)",
        ax = 0,
        ay = 55,
        bgcolor = "rgba(255, 255, 255, 0.92)",
        bordercolor = "rgba(15, 23, 42, 0.15)",
        borderwidth = 1,
        borderpad = 5,
        font = {"size": 11},
        row = 1,
        col = 1,
    )
    fig.add_hline(
        y = 1,
        row = 2,
        col = 1,
        line_width = 1.0,
        line_color = "rgba(15, 23, 42, 0.45)",
    )
    fig.update_layout(
        title = {
            "text": (
                "<b>Forecast-Based Battery Strategy Performance</b>"
                "<br>"
                "<sup>"
                f"2026 holdout backtest • "
                f"Net profit: ${total_profit:,.0f} • "
                f"Maximum drawdown: ${max_drawdown:,.0f}"
                "</sup>"
            ),
            "x": 0.045,
            "xanchor": "left",
            "y": 0.975,
            "yanchor": "top",
        },
        template = "plotly_white",
        height = 760,
        hovermode = "x unified",
        margin = {"l": 90, "r": 55, "t": 105, "b": 75},
        showlegend = False,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
    )
    fig.update_yaxes(
        title_text = "Cumulative Profit ($)",
        tickprefix = "$",
        tickformat = "~s",
        gridcolor = "rgba(15, 23, 42, 0.08)",
        zeroline = False,
        row = 1,
        col = 1
    )
    fig.update_yaxes(
        title_text = "Drawdown ($)",
        tickprefix = "$",
        tickformat = "~s",
        gridcolor = "rgba(15, 23, 42, 0.08)",
        zeroline = False,
        row = 2,
        col = 1,
    )
    fig.update_xaxes(
        showgrid = False,
        tickformat = "%b %Y",
        title_text = "Date",
        row = 2,
        col = 1,
    )
    _save_figure(fig, output_stem)
    return daily

def plot_strategy_performance_comparison(
    rules_summary: dict,
    optimization_summary: dict,
    ppo_summary: dict,
    output_stem: str = "outputs/figures/strategy_performance_comparison",
) -> pd.DataFrame:
    required_metrics = [
        "total_profit_usd",
        "annualized_daily_sharpe",
        "maximum_drawdown_usd",
        "profit_factor",
        "equivalent_full_cycles",
    ]
    summaries = {
        "Rules-Based": rules_summary,
        "CVXPY": optimization_summary,
        "PPO": ppo_summary,
    }

    comparison = pd.DataFrame(
        {
            "Strategy": [
                "Rules-Based",
                "CVXPY",
                "PPO",
            ],
            "Total Profit": [
                rules_summary["total_profit_usd"],
                optimization_summary["total_profit_usd"],
                ppo_summary["total_profit_usd"],
            ],
            "Sharpe Ratio": [
                rules_summary["annualized_daily_sharpe"],
                optimization_summary["annualized_daily_sharpe"],
                ppo_summary["annualized_daily_sharpe"],
            ],
            "Maximum Drawdown": [
                rules_summary["maximum_drawdown_usd"],
                optimization_summary["maximum_drawdown_usd"],
                ppo_summary["maximum_drawdown_usd"],
            ],
            "Profit Factor": [
                rules_summary["profit_factor"],
                optimization_summary["profit_factor"],
                ppo_summary["profit_factor"],
            ],
            "Equivalent Full Cycles": [
                rules_summary["equivalent_full_cycles"],
                optimization_summary["equivalent_full_cycles"],
                ppo_summary["equivalent_full_cycles"],
            ],
        }
    )

    strategy_order = [
        "Rules-Based",
        "CVXPY",
        "PPO",
    ]
    strategy_colors = {
        "Rules-Based": "#94A3B8",
        "CVXPY": "#2563EB",
        "PPO": "#16A34A",
    }
    metric_config = [
        {
            "column": "Total Profit",
            "title": "Total Net Profit",
            "axis_title": "Profit ($)",
            "tickprefix": "$",
            "tickformat": "~s",
            "value_format": "${:,.0f}",
            "higher_is_better": True,
        },
        {
            "column": "Sharpe Ratio",
            "title": "Annualized Daily Sharpe Ratio",
            "axis_title": "Sharpe Ratio",
            "tickprefix": "",
            "tickformat": ".1f",
            "value_format": "{:.2f}",
            "higher_is_better": True,
        },
        {
            "column": "Maximum Drawdown",
            "title": "Maximum Drawdown",
            "axis_title": "Drawdown ($)",
            "tickprefix": "$",
            "tickformat": "~s",
            "value_format": "${:,.0f}",
            "higher_is_better": False,
        },
        {
            "column": "Profit Factor",
            "title": "Profit Factor",
            "axis_title": "Profit Factor",
            "tickprefix": "",
            "tickformat": ".1f",
            "value_format": "{:.2f}",
            "higher_is_better": True,
        },
        {
            "column": "Equivalent Full Cycles",
            "title": "Equivalent Full Cycles",
            "axis_title": "Cycles",
            "tickprefix": "",
            "tickformat": ".0f",
            "value_format": "{:.1f}",
            "higher_is_better": None,
        },
    ]
    subplot_titles = [
        (
            f"<b>{metric["title"]}</b>"
            + (
                "<br><sup>Higher is better</sup>"
                if metric["higher_is_better"] is True
                else "<br><sup>Lower is better</sup>"
                if metric["higher_is_better"] is False
                else "<br><sup>Operational utilization</sup>"
            )
        )
        for metric in metric_config
    ]
    fig = make_subplots(
        rows = 5,
        cols = 1,
        vertical_spacing = 0.075,
        subplot_titles = subplot_titles,
        row_heights = [0.20, 0.20, 0.20, 0.20, 0.20],
    )

    for row_number, metric in enumerate(metric_config, start = 1):
        metric_values = comparison.set_index("Strategy").loc[
            strategy_order,
            metric["column"],
        ]
        maximum_value = metric_values.max()

        if maximum_value > 0:
            axis_upper_bound = maximum_value * 1.22
        else:
            axis_upper_bound = 1
        
        for strategy in strategy_order:
            value = comparison.loc[
                comparison["Strategy"] == strategy,
                metric["column"],
            ].iloc[0]

            fig.add_trace(
                go.Bar(
                    x = [value],
                    y = [strategy],
                    orientation = "h",
                    name = strategy,
                    marker = {
                        "color": strategy_colors[strategy],
                        "line": {
                            "color": "rgba(15, 23, 42, 0.10)",
                            "width": 0.7,
                        },
                    },
                    width = 0.58,
                    text = [metric["value_format"].format(value)],
                    textposition = "outside",
                    textfont = {
                        "size": 11,
                        "color": "#0F172A",
                    },
                    cliponaxis = False,
                    hovertemplate = (
                        f"<b>{strategy}</b><br>"
                        f"{metric["title"]}: "
                        f"{metric["value_format"].format(value)}"
                        "<extra></extra>"
                    ),
                    showlegend = False,
                ),
                row = row_number,
                col = 1,
            )
        fig.update_xaxes(
            title_text = metric["axis_title"],
            range = [0, axis_upper_bound],
            tickprefix = metric["tickprefix"],
            tickformat = metric["tickformat"],
            gridcolor = "rgba(15, 23, 42, 0.08)",
            zeroline = False,
            showline = False,
            row = row_number,
            col = 1,
        )
        fig.update_yaxes(
            categoryorder = "array",
            categoryarray = list(reversed(strategy_order)),
            showgrid = False,
            tickfont = {"size": 11, "color": "#334155"},
            row = row_number,
            col = 1,
        )
    
    total_profit_values = comparison.set_index("Strategy")["Total Profit"]
    best_profit_strategy = total_profit_values.idxmax()
    best_profit_value = total_profit_values.max()

    sharpe_values = comparison.set_index("Strategy")["Sharpe Ratio"]
    best_sharpe_strategy = sharpe_values.idxmax()
    best_sharpe_value = sharpe_values.max()

    drawdown_values = comparison.set_index("Strategy")["Maximum Drawdown"]
    lowest_drawdown_strategy = drawdown_values.idxmin()
    lowest_drawdown_value = drawdown_values.min()

    fig.update_layout(
        title = {
            "text": (
                "<b>Operational Performance Comparison Across "
                "Battery Dispatch Strategies</b>"
                "<br>"
                "<sup>"
                "2026 holdout backtest • "
                "Deployable rules-based, optimization, and "
                "reinforcement-learning strategies"
                "</sup>"
            ),
            "x": 0.045,
            "xanchor": "left",
            "y": 0.988,
            "yanchor": "top",
        },
        template = "plotly_white",
        height = 1220,
        margin = {"l": 135, "r": 105, "t": 125, "b": 90},
        showlegend = False,
        bargap = 0.30,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        font = {
            "family": "Arial",
            "size": 12,
            "color": "#0F172A",
        },
    )
    fig.update_annotations(
        font = {
            "size": 13,
            "color": "#0F172A",
        },
        x = 0,
        xanchor = "left",
    )
    fig.add_annotation(
        x = 0,
        y = -0.085,
        xref = "paper",
        yref = "paper",
        text = (
            f"<b>Performance summary:</b> "
            f"{best_profit_strategy} generated the highest total profit "
            f"(${best_profit_value:,.0f}); "
            f"{best_sharpe_strategy} achieved the highest Sharpe ratio "
            f"({best_sharpe_value:.2f}); "
            f"{lowest_drawdown_strategy} recorded the lowest maximum drawdown "
            f"(${lowest_drawdown_value:,.0f})."
        ),
        showarrow = False,
        align = "left",
        xanchor = "left",
        font = {
            "size": 11,
            "color": "#475569",
        }
    )
    _save_figure(fig, output_stem)
    return comparison

def plot_ppo_action_price_distribution(
    ppo_dispatch: pd.DataFrame,
    price_column: str = "Forecast Price",
    action_column: str = "Action Name",
    output_stem: str = "outputs/figures/ppo_action_price_distribution",
    action_mapping: dict | None = None,
) -> pd.DataFrame:
    data = ppo_dispatch[
        [
            price_column,
            action_column,
        ]
    ].copy()

    data = data.rename(
        columns = {
            price_column: "Forecast Price",
            action_column: "Action",
        }
    )

    data["Forecast Price"] = pd.to_numeric(
        data["Forecast Price"],
        errors = "coerce",
    )

    # Standardize text labels in case capitalization differs.
    text_action_mapping = {
        "hold": "Hold",
        "idle": "Hold",
        "charge": "Charge",
        "charging": "Charge",
        "discharge": "Discharge",
        "discharging": "Discharge",
    }

    data["Action"] = (
        data["Action"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(text_action_mapping)
    )

    data = data.dropna(
        subset = [
            "Forecast Price",
            "Action",
        ]
    ).reset_index(drop = True)

    action_order = [
        "Charge",
        "Hold",
        "Discharge",
    ]

    available_actions = [
        action
        for action in action_order
        if action in data["Action"].unique()
    ]

    if len(available_actions) < 2:
        raise ValueError(
            "Fewer than two valid PPO actions were found after cleaning. "
            "Check the action column and action_mapping."
        )

    strategy_colors = {
        "Charge": "#2563EB",
        "Hold": "#94A3B8",
        "Discharge": "#16A34A",
    }

    summary = (
        data.groupby("Action")
        .agg(
            Intervals = ("Forecast Price", "size"),
            Mean_Price = ("Forecast Price", "mean"),
            Median_Price = ("Forecast Price", "median"),
            Price_Std = ("Forecast Price", "std"),
            Minimum_Price = ("Forecast Price", "min"),
            Maximum_Price = ("Forecast Price", "max"),
        )
        .reindex(available_actions)
        .reset_index()
    )

    summary["Action Percentage"] = (
        summary["Intervals"]
        / summary["Intervals"].sum()
        * 100
    )

    # Winsorize only for visualization so extreme ERCOT price spikes do not
    # flatten the central distributions into decorative pancakes.
    lower_visual_bound = data["Forecast Price"].quantile(0.01)
    upper_visual_bound = data["Forecast Price"].quantile(0.99)

    plot_data = data.loc[
        data["Forecast Price"].between(
            lower_visual_bound,
            upper_visual_bound,
        )
    ].copy()

    fig = go.Figure()

    for action in available_actions:
        action_data = plot_data.loc[
            plot_data["Action"] == action,
            "Forecast Price",
        ]

        action_summary = summary.loc[
            summary["Action"] == action
        ].iloc[0]

        fig.add_trace(
            go.Violin(
                x = action_data,
                y = [action] * len(action_data),
                name = action,
                orientation = "h",
                line = {
                    "color": strategy_colors[action],
                    "width": 1.8,
                },
                fillcolor = {
                    "Charge": "rgba(37, 99, 235, 0.18)",
                    "Hold": "rgba(148, 163, 184, 0.20)",
                    "Discharge": "rgba(22, 163, 74, 0.18)",
                }[action],
                meanline_visible = False,
                box_visible = True,
                points = False,
                spanmode = "hard",
                width = 0.78,
                customdata = np.column_stack(
                    [
                        np.full(
                            len(action_data),
                            action_summary["Median_Price"],
                        ),
                        np.full(
                            len(action_data),
                            action_summary["Action Percentage"],
                        ),
                        np.full(
                            len(action_data),
                            action_summary["Intervals"],
                        ),
                    ]
                ),
                hovertemplate = (
                    f"<b>{action}</b><br>"
                    "Forecast price: $%{x:,.2f}/MWh<br>"
                    "Median price: $%{customdata[0]:,.2f}/MWh<br>"
                    "Action share: %{customdata[1]:.1f}%<br>"
                    "Intervals: %{customdata[2]:,.0f}"
                    "<extra></extra>"
                ),
                showlegend = False,
            )
        )

    for action in available_actions:
        action_summary = summary.loc[
            summary["Action"] == action
        ].iloc[0]

        fig.add_annotation(
            x = action_summary["Median_Price"],
            y = action,
            text = (
                f"<b>${action_summary['Median_Price']:,.2f}</b>"
                f"<br>"
                f"<span style='font-size:10px'>"
                f"{action_summary['Action Percentage']:.1f}% of intervals"
                f"</span>"
            ),
            showarrow = True,
            arrowhead = 2,
            arrowsize = 1,
            arrowwidth = 1,
            arrowcolor = "rgba(15, 23, 42, 0.50)",
            ax = 45,
            ay = -28,
            bgcolor = "rgba(255, 255, 255, 0.94)",
            bordercolor = "rgba(15, 23, 42, 0.14)",
            borderwidth = 1,
            borderpad = 5,
            font = {
                "size": 11,
                "color": "#0F172A",
            },
        )

    charge_median = (
        summary.loc[
            summary["Action"] == "Charge",
            "Median_Price",
        ].iloc[0]
        if "Charge" in available_actions
        else np.nan
    )

    hold_median = (
        summary.loc[
            summary["Action"] == "Hold",
            "Median_Price",
        ].iloc[0]
        if "Hold" in available_actions
        else np.nan
    )

    discharge_median = (
        summary.loc[
            summary["Action"] == "Discharge",
            "Median_Price",
        ].iloc[0]
        if "Discharge" in available_actions
        else np.nan
    )

    if (
        pd.notna(charge_median)
        and pd.notna(discharge_median)
    ):
        median_spread = discharge_median - charge_median

        subtitle_text = (
            "2026 holdout backtest • "
            f"Median discharge-charge forecast spread: "
            f"${median_spread:,.2}/MWh"
        )
    else:
        subtitle_text = (
            "2026 holdout backtest • "
            "Forecast price distributions by PPO dispatch action"
        )

    fig.update_layout(
        title = {
            "text": (
                "<b>PPO Policy Actions Across Forecasted Electricity Prices</b>"
                "<br>"
                "<sup>"
                f"{subtitle_text}"
                "</sup>"
            ),
            "x": 0.045,
            "xanchor": "left",
            "y": 0.975,
            "yanchor": "top",
        },
        template = "plotly_white",
        height = 620,
        margin = {
            "l": 105,
            "r": 80,
            "t": 110,
            "b": 80,
        },
        showlegend = False,
        violinmode = "overlay",
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        font = {
            "family": "Arial",
            "size": 12,
            "color": "#0F172A",
        },
    )

    fig.update_xaxes(
        title_text = "Forecasted Electricity Price ($/MWh)",
        tickprefix = "$",
        tickformat = ",.0f",
        gridcolor = "rgba(15, 23, 42, 0.08)",
        zeroline = True,
        zerolinewidth = 1,
        zerolinecolor = "rgba(15, 23, 42, 0.25)",
        range = [
            lower_visual_bound,
            upper_visual_bound,
        ],
    )

    fig.update_yaxes(
        title_text = "PPO Action",
        categoryorder = "array",
        categoryarray = list(reversed(available_actions)),
        showgrid = False,
        tickfont = {
            "size": 12,
            "color": "#334155",
        },
    )

    fig.add_annotation(
        x = 0,
        y = -0.17,
        xref = "paper",
        yref = "paper",
        text = (
            "Violin widths represent relative price density. "
            "Box plots show the interquartile range and median. "
            "The displayed range excludes the most extreme 1% of prices "
            "on each tail for readability; summary statistics use all intervals."
        ),
        showarrow = False,
        xanchor = "left",
        align = "left",
        font = {
            "size": 10,
            "color": "#64748B",
        },
    )

    _save_figure(fig, output_stem)

    return summary

def plot_ppo_soc_heatmap(
    ppo_dispatch: pd.DataFrame,
    timestamp_column: str = "time_stamp",
    soc_column: str = "SOC MWh",
    battery_capacity_mwh = BATTERY_CAPACITY_MWH,
    output_stem: str = "outputs/figures/ppo_soc_operating_profile"
) -> pd.DataFrame:
    data = ppo_dispatch[[timestamp_column, soc_column]].copy()
    data = data.rename(
        columns={
            timestamp_column: "time_stamp",
            soc_column: "State of Charge",
        }
    )

    data["time_stamp"] = pd.to_datetime(
        data["time_stamp"],
        errors="coerce",
    )
    data["State of Charge"] = pd.to_numeric(
        data["State of Charge"],
        errors="coerce",
    )

    data = (
        data.dropna(subset=["time_stamp", "State of Charge"])
        .sort_values("time_stamp")
        .reset_index(drop=True)
    )

    data["month_number"] = data["time_stamp"].dt.month
    data["month_label"] = data["time_stamp"].dt.strftime("%b")
    data["hour"] = data["time_stamp"].dt.hour

    monthly_hourly_soc = (
        data.groupby(
            ["month_number", "month_label", "hour"],
            as_index=False,
        )
        .agg(
            Average_SOC_MWh=("State of Charge", "mean"),
            Median_SOC_MWh=("State of Charge", "median"),
            Minimum_SOC_MWh=("State of Charge", "min"),
            Maximum_SOC_MWh=("State of Charge", "max"),
            Observations=("State of Charge", "size"),
        )
        .sort_values(["month_number", "hour"])
        .reset_index(drop=True)
    )

    monthly_hourly_soc["Average_SOC_Percentage"] = (
        monthly_hourly_soc["Average_SOC_MWh"]
        / battery_capacity_mwh
        * 100
    )

    month_order = (
        monthly_hourly_soc[["month_number", "month_label"]]
        .drop_duplicates()
        .sort_values("month_number")["month_label"]
        .tolist()
    )
    hour_order = list(range(24))

    soc_pivot = (
        monthly_hourly_soc.pivot(
            index="month_label",
            columns="hour",
            values="Average_SOC_MWh",
        )
        .reindex(index=month_order, columns=hour_order)
    )

    soc_percentage_pivot = (
        monthly_hourly_soc.pivot(
            index="month_label",
            columns="hour",
            values="Average_SOC_Percentage",
        )
        .reindex(index=month_order, columns=hour_order)
    )

    observation_pivot = (
        monthly_hourly_soc.pivot(
            index="month_label",
            columns="hour",
            values="Observations",
        )
        .reindex(index=month_order, columns=hour_order)
    )

    average_soc = data["State of Charge"].mean()
    minimum_soc = data["State of Charge"].min()
    maximum_soc = data["State of Charge"].max()

    hourly_average_soc = (
        data.groupby("hour")["State of Charge"]
        .mean()
        .reindex(hour_order)
    )

    highest_soc_hour = int(hourly_average_soc.idxmax())
    lowest_soc_hour = int(hourly_average_soc.idxmin())
    highest_soc_value = float(hourly_average_soc.loc[highest_soc_hour])
    lowest_soc_value = float(hourly_average_soc.loc[lowest_soc_hour])

    customdata = np.stack(
        [
            soc_percentage_pivot.to_numpy(),
            observation_pivot.to_numpy(),
        ],
        axis=-1,
    )

    colorscale = [
        [0.00, "#F8FAFC"],
        [0.15, "#EFF6FF"],
        [0.35, "#BFDBFE"],
        [0.55, "#60A5FA"],
        [0.75, "#2563EB"],
        [1.00, "#1E3A8A"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=hour_order,
            y=month_order,
            z=soc_pivot.to_numpy(),
            customdata=customdata,
            colorscale=colorscale,
            zmin=0,
            zmax=battery_capacity_mwh,
            colorbar={
                "title": {
                    "text": "Average SOC<br>(MWh)",
                    "side": "right",
                },
                "tickformat": ".0f",
                "thickness": 14,
                "len": 0.78,
                "outlinewidth": 0,
                "x": 1.02,
                "xpad": 12,
                "tickfont": {
                    "size": 10,
                    "color": "#475569",
                },

            },
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Hour: %{x}:00<br>"
                "Average SOC: %{z:,.1f} MWh<br>"
                "Average SOC: %{customdata[0]:.1f}%<br>"
                "Observations: %{customdata[1]:,.0f}"
                "<extra></extra>"
            ),
            xgap=1.2,
            ygap=1.2,
        )
    )

    fig.add_vline(
        x=highest_soc_hour,
        line_width=1.2,
        line_dash="dot",
        line_color="rgba(37, 99, 235, 0.55)",
    )
    fig.add_vline(
        x=lowest_soc_hour,
        line_width=1.2,
        line_dash="dot",
        line_color="rgba(15, 23, 42, 0.45)",
    )

    fig.add_annotation(
        x=highest_soc_hour,
        y=1.045,
        xref="x",
        yref="paper",
        text=(
            f"<b>Highest average SOC</b><br>"
            f"{highest_soc_hour:02d}:00 • {highest_soc_value:,.1f} MWh"
        ),
        showarrow=False,
        align="center",
        font={"size": 10, "color": "#2563EB"},
        bgcolor="rgba(255, 255, 255, 0.90)",
        bordercolor="rgba(37, 99, 235, 0.18)",
        borderwidth=1,
        borderpad=4,
    )

    fig.add_annotation(
        x=lowest_soc_hour,
        y=-0.17,
        xref="x",
        yref="paper",
        text=(
            f"<b>Lowest average SOC</b><br>"
            f"{lowest_soc_hour:02d}:00 • {lowest_soc_value:,.1f} MWh"
        ),
        showarrow=False,
        align="center",
        font={"size": 10, "color": "#475569"},
        bgcolor="rgba(255, 255, 255, 0.90)",
        bordercolor="rgba(15, 23, 42, 0.14)",
        borderwidth=1,
        borderpad=4,
    )

    fig.update_layout(
        title={
            "text": (
                "<b>PPO Battery State-of-Charge Operating Profile</b>"
                "<br>"
                "<sup>"
                "2026 holdout backtest • "
                f"Average SOC: {average_soc:,.1f} MWh • "
                f"Observed range: {minimum_soc:,.1f} - {maximum_soc:,.1f} MWh"
                "</sup>"
            ),
            "x": 0.045,
            "xanchor": "left",
            "y": 0.975,
            "yanchor": "top",
        },
        template="plotly_white",
        height=650,
        margin={"l": 95, "r": 120, "t": 115, "b": 105},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={
            "family": "Arial",
            "size": 12,
            "color": "#0F172A",
        },
    )

    fig.update_xaxes(
        title_text="Hour of Day",
        tickmode="array",
        tickvals=[0, 3, 6, 9, 12, 15, 18, 21, 23],
        ticktext=[
            "00:00",
            "03:00",
            "06:00",
            "09:00",
            "12:00",
            "15:00",
            "18:00",
            "21:00",
            "23:00",
        ],
        range=[-0.5, 23.5],
        showgrid=False,
        zeroline=False,
        tickfont={"size": 11, "color": "#475569"},
    )

    fig.update_yaxes(
        title_text="Month",
        categoryorder="array",
        categoryarray=list(reversed(month_order)),
        showgrid=False,
        zeroline=False,
        tickfont={"size": 11, "color": "#475569"},
    )

    _save_figure(fig, output_stem)
    return soc_pivot

def plot_forecast_horizon_performance_comparison(
    rolling_4h_summary: Mapping[str, Any],
    horizon_24h_summary: Mapping[str, Any],
    perfect_foresight_summary: Mapping[str, Any],
    output_stem: str = "outputs/figures/forecast_horizon_performance_comparison",
) -> pd.DataFrame:

    strategy_names = [
        "4-Hour Rolling",
        "24-Hour Horizon",
        "Perfect Foresight",
    ]

    summaries = [
        rolling_4h_summary,
        horizon_24h_summary,
        perfect_foresight_summary,
    ]

    metric_keys = {
        "Total Net Profit": "total_profit_usd",
        "Annualized Daily Sharpe Ratio": "annualized_daily_sharpe",
        "Maximum Drawdown": "maximum_drawdown_usd",
        "Profit Factor": "profit_factor",
        "Equivalent Full Cycles": "equivalent_full_cycles",
    }


    comparison = pd.DataFrame(
        {
            "Forecast Horizon": strategy_names,
            "Total Net Profit": [
                float(summary["total_profit_usd"])
                for summary in summaries
            ],
            "Annualized Daily Sharpe Ratio": [
                float(summary["annualized_daily_sharpe"])
                for summary in summaries
            ],
            "Maximum Drawdown": [
                abs(float(summary["maximum_drawdown_usd"]))
                for summary in summaries
            ],
            "Profit Factor": [
                float(summary["profit_factor"])
                for summary in summaries
            ],
            "Equivalent Full Cycles": [
                float(summary["equivalent_full_cycles"])
                for summary in summaries
            ],
        }
    )

    colors = {
        "4-Hour Rolling": "#2563EB",
        "24-Hour Horizon": "#1D4ED8",
        "Perfect Foresight": "#D4A72C",
    }

    metric_order = [
        "Total Net Profit",
        "Annualized Daily Sharpe Ratio",
        "Maximum Drawdown",
        "Profit Factor",
        "Equivalent Full Cycles",
    ]

    subtitles = {
        "Total Net Profit": "Higher is better",
        "Annualized Daily Sharpe Ratio": "Higher is better",
        "Maximum Drawdown": "Lower is better",
        "Profit Factor": "Higher is better",
        "Equivalent Full Cycles": "Battery utilization",
    }

    value_formats = {
        "Total Net Profit": lambda value: f"${value:,.0f}",
        "Annualized Daily Sharpe Ratio": lambda value: f"{value:.2f}",
        "Maximum Drawdown": lambda value: f"${value:,.0f}",
        "Profit Factor": lambda value: f"{value:.2f}",
        "Equivalent Full Cycles": lambda value: f"{value:,.1f}",
    }

    hover_templates = {
        "Total Net Profit": (
            "<b>%{y}</b><br>"
            "Total Net Profit: $%{x:,.0f}"
            "<extra></extra>"
        ),
        "Annualized Daily Sharpe Ratio": (
            "<b>%{y}</b><br>"
            "Annualized Daily Sharpe: %{x:.2f}"
            "<extra></extra>"
        ),
        "Maximum Drawdown": (
            "<b>%{y}</b><br>"
            "Maximum Drawdown: $%{x:,.0f}"
            "<extra></extra>"
        ),
        "Profit Factor": (
            "<b>%{y}</b><br>"
            "Profit Factor: %{x:.2f}"
            "<extra></extra>"
        ),
        "Equivalent Full Cycles": (
            "<b>%{y}</b><br>"
            "Equivalent Full Cycles: %{x:,.1f}"
            "<extra></extra>"
        ),
    }

    fig = make_subplots(
        rows = 5,
        cols = 1,
        vertical_spacing = 0.075,
        subplot_titles = [
            (
                f"<b>{metric}</b>"
                f"<br><span style='font-size:10px;color:#64748B'>"
                f"{subtitles[metric]}"
                f"</span>"
            )
            for metric in metric_order
        ],
    )

    for row_number, metric in enumerate(metric_order, start = 1):
        values = comparison[metric].tolist()

        max_value = max(values) if values else 0
        axis_padding = max_value * 0.22 if max_value > 0 else 1

        fig.add_trace(
            go.Bar(
                x = values,
                y = strategy_names,
                orientation = "h",
                marker = {
                    "color": [
                        colors[strategy]
                        for strategy in strategy_names
                    ],
                    "line": {
                        "width": 0,
                    },
                },
                text = [
                    value_formats[metric](value)
                    for value in values
                ],
                textposition = "outside",
                textfont = {
                    "size": 11,
                    "color": "#0F172A",
                },
                cliponaxis = False,
                hovertemplate = hover_templates[metric],
                showlegend = False,
            ),
            row = row_number,
            col = 1,
        )

        fig.update_xaxes(
            range = [0, max_value + axis_padding],
            showgrid = True,
            gridcolor = "rgba(148, 163, 184, 0.18)",
            gridwidth = 1,
            zeroline = False,
            showticklabels = False,
            ticks = "",
            fixedrange = True,
            row = row_number,
            col = 1,
        )

        fig.update_yaxes(
            categoryorder = "array",
            categoryarray = list(reversed(strategy_names)),
            showgrid = False,
            zeroline = False,
            tickfont = {
                "size": 11,
                "color": "#334155",
            },
            fixedrange = True,
            row = row_number,
            col = 1,
        )

    fig.update_annotations(
        font = {
            "family": "Arial",
            "size": 12,
            "color": "#0F172A",
        },
        x = 0.0,
        xanchor = "left",
    )

    fig.update_layout(
        title = {
            "text": (
                "<b>Forecast Horizon Performance Comparison</b>"
                "<br>"
                "<sup>"
                "Battery arbitrage results across increasing levels of "
                "future price information"
                "</sup>"
            ),
            "x": 0.0275,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
        template = "plotly_white",
        height = 1050,
        margin = {
            "l": 150,
            "r": 85,
            "t": 120,
            "b": 55,
        },
        paper_bgcolor = "white",
        plot_bgcolor = "white",
        font = {
            "family": "Arial",
            "size": 12,
            "color": "#0F172A",
        },
        bargap = 0.35,
    )

    fig.add_annotation(
        x = 1.0,
        y = 1.035,
        xref = "paper",
        yref = "paper",
        text = (
            "<span style='color:#2563EB'> • </span> 4-Hour Rolling"
            "&nbsp;&nbsp;&nbsp;"
            "<span style='color:#1D4ED8'> • </span> 24-Hour Horizon"
            "&nbsp;&nbsp;&nbsp;"
            "<span style='color:#D4A72C'> •</span> Perfect Foresight"
        ),
        showarrow = False,
        xanchor = "right",
        yanchor = "bottom",
        font = {
            "size": 11,
            "color": "#475569",
        },
    )

    fig.add_annotation(
        x = 0,
        y = -0.035,
        xref = "paper",
        yref = "paper",
        text = (
            "Perfect foresight is a theoretical upper bound, not a deployable "
            "strategy. Maximum drawdown is shown as an absolute loss magnitude."
        ),
        showarrow = False,
        xanchor = "left",
        yanchor = "top",
        align = "left",
        font = {
            "size": 10,
            "color": "#64748B",
        },
    )

    _save_figure(
        fig,
        output_stem,
    )

    return comparison

def plot_forecast_horizon_cumulative_profit(
    rolling_4h_results: pd.DataFrame,
    horizon_24h_results: pd.DataFrame,
    perfect_foresight_results: pd.DataFrame,
    timestamp_column: str = "time_stamp",
    cumulative_profit_column: str = "Cumulative Profit",
    output_stem: str = "outputs/figures/forecast_horizon_cumulative_profit",
    title: str = "Forecast Horizon Cumulative Profit",
    subtitle: str = (
        "Battery arbitrage performance across increasing levels of future "
        "price information"
    ),
    add_endpoint_annotations: bool = True,
    add_peak_markers: bool = True,
) -> pd.DataFrame:

    strategy_inputs = {
        "4-Hour Rolling": rolling_4h_results,
        "24-Hour Horizon": horizon_24h_results,
        "Perfect Foresight": perfect_foresight_results,
    }

    cleaned_frames = {}

    for strategy_name, frame in strategy_inputs.items():
        required = [timestamp_column, cumulative_profit_column]
        missing = [column for column in required if column not in frame.columns]

        if missing:
            raise KeyError(
                f"{strategy_name} results are missing required columns: "
                f"{missing}. Available columns: {list(frame.columns)}"
            )

        cleaned = frame[required].copy().rename(
            columns={
                timestamp_column: "time_stamp",
                cumulative_profit_column: strategy_name,
            }
        )

        cleaned["time_stamp"] = pd.to_datetime(
            cleaned["time_stamp"],
            errors="coerce",
        )
        cleaned[strategy_name] = pd.to_numeric(
            cleaned[strategy_name],
            errors="coerce",
        )

        cleaned = (
            cleaned.dropna(subset=["time_stamp", strategy_name])
            .sort_values("time_stamp")
            .drop_duplicates(subset=["time_stamp"], keep="last")
            .reset_index(drop=True)
        )

        if cleaned.empty:
            raise ValueError(
                f"{strategy_name} contains no valid observations after cleaning."
            )

        cleaned_frames[strategy_name] = cleaned

    comparison = cleaned_frames["4-Hour Rolling"].merge(
        cleaned_frames["24-Hour Horizon"],
        on="time_stamp",
        how="outer",
    ).merge(
        cleaned_frames["Perfect Foresight"],
        on="time_stamp",
        how="outer",
    )
    comparison["time_stamp"] = comparison["time_stamp"].dt.to_pydatetime()
    comparison = comparison.sort_values("time_stamp").reset_index(drop=True)

    strategy_columns = [
        "4-Hour Rolling",
        "24-Hour Horizon",
        "Perfect Foresight",
    ]

    comparison[strategy_columns] = (
        comparison[strategy_columns]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    colors = {
        "4-Hour Rolling": "#2563EB",
        "24-Hour Horizon": "#1D4ED8",
        "Perfect Foresight": "#D4A72C",
    }

    fig = go.Figure()

    for strategy_name in strategy_columns:
        fig.add_trace(
            go.Scatter(
                x=comparison["time_stamp"],
                y=comparison[strategy_name],
                mode="lines",
                name=strategy_name,
                line={
                    "color": colors[strategy_name],
                    "width": 3.0 if strategy_name == "Perfect Foresight" else 2.7,
                },
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{x|%b %d, %Y %H:%M}<br>"
                    "Cumulative Profit: $%{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )

    if add_peak_markers:
        for strategy_name in strategy_columns:
            peak_index = comparison[strategy_name].idxmax()
            peak_time = comparison.loc[peak_index, "time_stamp"]
            peak_value = comparison.loc[peak_index, strategy_name]

            fig.add_trace(
                go.Scatter(
                    x=[peak_time],
                    y=[peak_value],
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": colors[strategy_name],
                        "line": {"color": "white", "width": 1.5},
                    },
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{strategy_name} Peak</b><br>"
                        "%{x|%b %d, %Y %H:%M}<br>"
                        "Peak Profit: $%{y:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

    final_time = comparison["time_stamp"].iloc[-1]
    final_values = {
        strategy_name: float(comparison[strategy_name].iloc[-1])
        for strategy_name in strategy_columns
    }

    if add_endpoint_annotations:
        sorted_endpoints = sorted(final_values.items(), key=lambda item: item[1])
        offsets = {
            strategy_name: offset
            for offset, (strategy_name, _) in zip([-28, 0, 14], sorted_endpoints)
        }

        for strategy_name in strategy_columns:
            fig.add_annotation(
                x=final_time,
                y=final_values[strategy_name],
                text=(
                    f"<b>{strategy_name}</b><br>"
                    f"${final_values[strategy_name]:,.0f}"
                ),
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor=colors[strategy_name],
                ax=82,
                ay=offsets[strategy_name],
                xanchor="left",
                align="left",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor=colors[strategy_name],
                borderwidth=1,
                borderpad=4,
                font={"size": 10, "color": "#0F172A"},
            )

    minimum_profit = float(comparison[strategy_columns].min().min())
    maximum_profit = float(comparison[strategy_columns].max().max())
    profit_range = maximum_profit - minimum_profit

    if profit_range <= 0:
        profit_range = max(abs(maximum_profit), 1.0)

    padding = profit_range * 0.10
    start_time = comparison["time_stamp"].min()
    end_time = comparison["time_stamp"].max()
    period_days = max((end_time - start_time).days, 1)

    fig.update_layout(
        title={
            "text": f"<b>{title}</b><br><sup>{subtitle}</sup>",
            "x": 0.035,
            "xanchor": "left",
            "y": 0.975,
            "yanchor": "top",
        },
        template="plotly_white",
        height=680,
        margin={
            "l": 95,
            "r": 175 if add_endpoint_annotations else 65,
            "t": 110,
            "b": 85,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial", "size": 12, "color": "#0F172A"},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 1.0,
            "xanchor": "right",
            "y": 1.035,
            "yanchor": "bottom",
            "font": {"size": 11, "color": "#475569"},
            "bgcolor": "rgba(255,255,255,0)",
            "borderwidth": 0,
        },
    )

    fig.update_xaxes(
        title_text="Holdout Backtest Date",
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="rgba(148,163,184,0.35)",
        linewidth=1,
        tickfont={"size": 11, "color": "#475569"},
        rangeslider_visible=False,
    )

    fig.update_yaxes(
        title_text="Cumulative Net Profit (USD)",
        range=[minimum_profit - padding, maximum_profit + padding],
        tickprefix="$",
        tickformat=",.0f",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="rgba(100,116,139,0.35)",
        zerolinewidth=1,
        showline=False,
        tickfont={"size": 11, "color": "#475569"},
    )

    fig.add_annotation(
        x=0,
        y=-0.14,
        xref="paper",
        yref="paper",
        text=(
            f"Backtest period: {start_time:%b %d, %Y} to "
            f"{end_time:%b %d, %Y} ({period_days:,} days). "
            "Perfect foresight is a theoretical upper bound rather than a "
            "deployable strategy."
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 10, "color": "#64748B"},
    )

    _save_figure(fig, output_stem)

    return comparison

def plot_battery_dispatch_duration_curve(
    dispatch_results: pd.DataFrame,
    net_dispatch_column: Optional[str] = None,
    charge_power_column: Optional[str] = None,
    discharge_power_column: Optional[str] = None,
    timestamp_column: Optional[str] = "time_stamp",
    idle_tolerance_mw: float = 1e-6,
    battery_power_mw: Optional[float] = None,
    output_stem: str = "outputs/figures/battery_dispatch_duration_curve",
) -> pd.DataFrame:

    if net_dispatch_column is None:
        if charge_power_column is None or discharge_power_column is None:
            raise ValueError(
                "Provide net_dispatch_column or both charge_power_column "
                "and discharge_power_column."
            )
        required = [charge_power_column, discharge_power_column]
    else:
        required = [net_dispatch_column]

    columns = list(required)
    has_timestamp = (
        timestamp_column is not None
        and timestamp_column in dispatch_results.columns
    )
    if has_timestamp:
        columns.append(timestamp_column)

    dispatch = dispatch_results[columns].copy()

    if has_timestamp:
        dispatch["time_stamp"] = pd.to_datetime(
            dispatch[timestamp_column],
            errors="coerce",
        )
    else:
        dispatch["time_stamp"] = pd.NaT

    if net_dispatch_column is not None:
        dispatch["net_dispatch_mw"] = pd.to_numeric(
            dispatch[net_dispatch_column],
            errors="coerce",
        )
    else:
        charge = pd.to_numeric(
            dispatch[charge_power_column],
            errors="coerce",
        )
        discharge = pd.to_numeric(
            dispatch[discharge_power_column],
            errors="coerce",
        )
        dispatch["net_dispatch_mw"] = discharge - charge

    dispatch = dispatch.replace([np.inf, -np.inf], np.nan)
    dispatch = dispatch.dropna(subset=["net_dispatch_mw"]).copy()

    if dispatch.empty:
        raise ValueError("No finite dispatch observations remain after cleaning.")

    dispatch["operating_state"] = np.select(
        [
            dispatch["net_dispatch_mw"] > idle_tolerance_mw,
            dispatch["net_dispatch_mw"] < -idle_tolerance_mw,
        ],
        ["Discharge", "Charge"],
        default="Idle",
    )

    duration = (
        dispatch.sort_values("net_dispatch_mw", ascending=False)
        .reset_index(drop=True)
    )

    n = len(duration)
    duration["duration_percentile"] = (
        0.0 if n == 1 else np.arange(n) / (n - 1) * 100.0
    )

    shares = (
        duration["operating_state"]
        .value_counts(normalize=True)
        .reindex(["Discharge", "Idle", "Charge"], fill_value=0.0)
        * 100.0
    )

    colors = {
        "Discharge": "#D4A72C",
        "Idle": "#94A3B8",
        "Charge": "#2563EB",
    }

    fig = go.Figure()

    for state in ["Discharge", "Idle", "Charge"]:
        state_data = duration[duration["operating_state"] == state]

        if state_data.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=state_data["duration_percentile"],
                y=state_data["net_dispatch_mw"],
                mode="lines",
                name=state,
                line={"color": colors[state], "width": 3},
                customdata=np.column_stack(
                    [
                        state_data["operating_state"],
                        state_data["time_stamp"]
                        .dt.strftime("%b %d, %Y %H:%M")
                        .fillna("Unavailable"),
                    ]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Duration Percentile: %{x:.2f}%<br>"
                    "Net Dispatch: %{y:,.2f} MW<br>"
                    "Original Interval: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=0,
        line_width=1.3,
        line_color="rgba(71,85,105,0.65)",
    )

    if battery_power_mw is not None and battery_power_mw > 0:
        for value, label in [
            (battery_power_mw, f"+{battery_power_mw:,.0f} MW limit"),
            (-battery_power_mw, f"-{battery_power_mw:,.0f} MW limit"),
        ]:
            fig.add_hline(
                y=value,
                line_width=1,
                line_dash="dot",
                line_color="rgba(100,116,139,0.5)",
                annotation_text=label,
                annotation_position="top right" if value > 0 else "bottom right",
                annotation_font={"size": 10, "color": "#64748B"},
            )

    mean_abs = float(duration["net_dispatch_mw"].abs().mean())

    fig.add_annotation(
        x=0.985,
        y=0.97,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Discharging:</b> {shares['Discharge']:.1f}%<br>"
            f"<b>Idle:</b> {shares['Idle']:.1f}%<br>"
            f"<b>Charging:</b> {shares['Charge']:.1f}%<br>"
            f"<b>Mean |dispatch|:</b> {mean_abs:,.1f} MW"
        ),
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="left",
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor="rgba(148,163,184,0.55)",
        borderwidth=1,
        borderpad=8,
        font={"size": 11, "color": "#0F172A"},
    )

    y_min = float(duration["net_dispatch_mw"].min())
    y_max = float(duration["net_dispatch_mw"].max())
    y_span = y_max - y_min
    if np.isclose(y_span, 0):
        y_span = max(abs(y_max), 1.0)

    fig.update_layout(
        title={
            "text": (
                "<b>Battery Dispatch Duration Curve</b><br>"
                "<sup>Distribution of charging, idle, and discharging behavior "
                "across the backtest period</sup>"
            ),
            "x": 0.035,
            "xanchor": "left",
            "y": 0.975,
            "yanchor": "top",
        },
        template="plotly_white",
        height=680,
        margin={"l": 95, "r": 65, "t": 110, "b": 95},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial", "size": 12, "color": "#0F172A"},
        hovermode="closest",
        legend={
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1.035,
            "yanchor": "bottom",
        },
    )

    fig.update_xaxes(
        title_text="Share of Backtest Intervals (%)",
        range=[0, 100],
        ticksuffix="%",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        showline=True,
        linecolor="rgba(148,163,184,0.35)",
    )

    fig.update_yaxes(
        title_text="Net Battery Dispatch (MW)",
        range=[y_min - 0.08 * y_span, y_max + 0.08 * y_span],
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
    )

    period_note = ""
    if duration["time_stamp"].notna().any():
        start = duration["time_stamp"].min()
        end = duration["time_stamp"].max()
        period_note = f" Backtest period: {start:%b %d, %Y} to {end:%b %d, %Y}."

    fig.add_annotation(
        x=0,
        y=-0.16,
        xref="paper",
        yref="paper",
        text=(
            "Dispatch is ranked from maximum discharge to maximum charge. "
            "Positive MW denotes grid injection; negative MW denotes charging."
            + period_note
        ),
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 10, "color": "#64748B"},
    )

    _save_figure(fig, output_stem)

    duration.attrs.update(
        {
            "observation_count": n,
            "discharge_share_percent": float(shares["Discharge"]),
            "idle_share_percent": float(shares["Idle"]),
            "charge_share_percent": float(shares["Charge"]),
            "peak_discharge_mw": y_max,
            "peak_charge_mw": y_min,
            "mean_absolute_dispatch_mw": mean_abs,
        }
    )

    return duration
