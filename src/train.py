# Training models
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from config import (RANDOM_STATE_CV, CV_SPLITS, N_JOBS_XGB, MAX_DEPTH_XGB, OBJECTIVE_XGB, SUBSAMPLE_XGB, N_ESTIMATORS_XGB, RANDOM_STATE_XGB, LEARNING_RATE_XGB, COLSAMPLE_BYTREE_XGB,
                    N_JOBS_LGBM, MAX_DEPTH_LGBM, SUBSAMPLE_LGBM, NUM_LEAVES_LGBM, N_ESTIMATORS_LGBM, RANDOM_STATE_LGBM, LEARNING_RATE_LGBM, COLSAMPLE_BYTREE_LGBM)
                    

def prepare_xy(df: pd.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df[features]
    y = df[target]
    return X, y

def train_linear_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators = 200, random_state = 42, n_jobs = -1)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators = N_ESTIMATORS_XGB,
        learning_rate = LEARNING_RATE_XGB,
        max_depth = MAX_DEPTH_XGB,
        subsample = SUBSAMPLE_XGB,
        colsample_bytree = COLSAMPLE_BYTREE_XGB,
        objective = OBJECTIVE_XGB,
        random_state = RANDOM_STATE_XGB,
        n_jobs = N_JOBS_XGB,
    )
    model.fit(X_train, y_train)
    return model

def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators = N_ESTIMATORS_LGBM,
        learning_rate = LEARNING_RATE_LGBM,
        max_depth = MAX_DEPTH_LGBM,
        num_leaves = NUM_LEAVES_LGBM,
        subsample = SUBSAMPLE_LGBM,
        colsample_bytree = COLSAMPLE_BYTREE_LGBM,
        random_state = RANDOM_STATE_LGBM,
        n_jobs = N_JOBS_LGBM,
    )
    model.fit(X_train, y_train)
    return model


tscv = TimeSeriesSplit(n_splits = CV_SPLITS)

def tune_xgboost(X_train, y_train):
    model = XGBRegressor(
        objective = "reg:squarederror",
        random_state = RANDOM_STATE_XGB,
        n_jobs = -1
    )

    param_grid = {
        "n_estimators": [300, 500, 700, 900, 1100],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "max_depth": [4, 5, 6, 8],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.45, 0.6, 0.75, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "gamma": [0, 0.1, 0.3, 0.5],
        "reg_alpha": [0.1, 1, 3, 5],
        "reg_lambda": [1, 5, 10, 15]
    }

    search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid,
        n_iter = 35,
        scoring = "neg_root_mean_squared_error",
        cv = tscv,
        verbose = 2,
        random_state = RANDOM_STATE_CV,
        n_jobs = 1
    )
    search.fit(X_train, y_train)
    print("Best Params:", search.best_params_)
    print("Best CV RMSE", search.best_score_)
    return search.best_estimator_

def tune_lightgbm(X_train, y_train):
    model = LGBMRegressor(
        random_state = RANDOM_STATE_LGBM,
        n_jobs = -1
    )

    param_grid = {
        "n_estimators": [300, 500, 700, 900, 1100],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "num_leaves": [12, 15, 23, 31, 45],
        "max_depth": [-1, 4, 5, 6],
        "min_child_samples": [15, 30, 50, 75],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.45, 0.6, 0.75, 0.9, 1.0],
        "reg_alpha": [0.1, 1, 3, 5],
        "reg_lambda": [1, 5, 10, 15]
    }

    search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid,
        n_iter = 35,
        scoring = "neg_root_mean_squared_error",
        cv = tscv,
        verbose = 2,
        random_state = RANDOM_STATE_CV,
        n_jobs = 1 # used to be -1, would time out
    )
    search.fit(X_train, y_train)
    print("Best Params:", search.best_params_)
    print("Best CV RMSE", search.best_score_)
    return search.best_estimator_
