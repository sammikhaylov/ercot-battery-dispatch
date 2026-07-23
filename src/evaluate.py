# RMSE / R2 / MAE
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions))
    }

def get_predictions(model, X_test: pd.DataFrame, y_test: pd.Series, timestamps: pd.Series) -> pd.DataFrame:
    predictions = model.predict(X_test)
    results = pd.DataFrame()
    results["time_stamp"] = pd.to_datetime(timestamps).to_numpy()
    results["Actual"] = y_test.to_numpy()
    results["Predicted"] = predictions
    results["Error"] = results["Actual"] - results["Predicted"]
    results["Absolute Error"] = abs(results["Error"])
    return results

def select_best_model(models: dict, metrics: dict, metric: str = "RMSE", lower_is_better: bool = True):
    metrics_df = pd.DataFrame([
        {"Model": name, **values}
        for name, values in metrics.items()
    ])
    metrics_df = metrics_df.sort_values(
        metric,
        ascending=lower_is_better
    ).reset_index(drop = True)
    best_model_name = metrics_df.loc[0, "Model"]
    best_model = models[best_model_name]
    return best_model_name, best_model, metrics_df