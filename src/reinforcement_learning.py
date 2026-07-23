from __future__ import annotations
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces

from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from config import (
    BATTERY_CAPACITY_MWH,
    MINIMUM_SOC_MWH,
    INITIAL_SOC_MWH,
    MAX_CHARGE_MW,
    MAX_DISCHARGE_MW,
    CHARGE_EFFICIENCY,
    DISCHARGE_EFFICIENCY,
    INTERVAL_HOURS,
    DEGRADATION_COST_PER_MWH,
    TRANSACTION_COST_PER_MWH,
)

class BatteryTradingEnv(gym.Env):
    """
    Observation:
    0: normalized forecast price
    1: normalized rolling mean
    2: normalized rolling volatility
    3: normalized state of charge
    4: hour sine
    5: hour cosine
    6: day of week sine
    7: day of week cosine

    Actions:
    0 = charge
    1 = hold
    2 = discharge

    Reward:
    realized_market_revenue - degradation_cost - transaction_cost
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        episode_length: int = 96 * 30, # One month
        random_start: bool = True,
        capacity_mwh: float = BATTERY_CAPACITY_MWH,
        minimum_soc_mwh: float = MINIMUM_SOC_MWH,
        initial_soc_mwh: float = INITIAL_SOC_MWH,
        max_charge_mw: float = MAX_CHARGE_MW,
        max_discharge_mw: float = MAX_DISCHARGE_MW,
        charge_efficiency: float = CHARGE_EFFICIENCY,
        discharge_efficiency: float = DISCHARGE_EFFICIENCY,
        interval_hours: float = INTERVAL_HOURS,
        degradation_cost_per_mwh: float = DEGRADATION_COST_PER_MWH,
        transaction_cost_per_mwh: float = TRANSACTION_COST_PER_MWH,
        reward_scale: float = 1_000.0,
    ) -> None:
        super().__init__()

        self.data = data.copy()
        self.data["time_stamp"] = pd.to_datetime(self.data["time_stamp"])
        self.data = self.data.sort_values("time_stamp").reset_index(drop = True)

        self.episode_length = min(int(episode_length), len(self.data))
        self.random_start = random_start
        self.capacity_mwh = float(capacity_mwh)
        self.minimum_soc_mwh = float(minimum_soc_mwh)
        self.initial_soc_mwh = float(initial_soc_mwh)
        self.max_charge_mw = float(max_charge_mw)
        self.max_discharge_mw = float(max_discharge_mw)
        self.charge_efficiency = float(charge_efficiency)
        self.discharge_efficiency = float(discharge_efficiency)
        self.interval_hours = float(interval_hours)
        self.degradation_cost_per_mwh = float(degradation_cost_per_mwh)
        self.transaction_cost_per_mwh = float(transaction_cost_per_mwh)
        self.reward_scale = float(reward_scale)
        self.history: list[dict] = []

        # Historical market context
        shifted_forecast = self.data["Predicted"].shift(1)

        self.data["Forecast Rolling Mean"] = shifted_forecast.rolling(window = 96, min_periods = 1).mean()
        self.data["Forecast Rolling Volatility"] = shifted_forecast.rolling(window = 96, min_periods = 2).std().fillna(0.0)
        self.data["Forecast Rolling Mean"] = self.data["Forecast Rolling Mean"].fillna(self.data["Predicted"].iloc[0])

        # normalized statistics calculated from environments data set, for final eval, construct from
        # training data and pass into train and test environments
        self.price_mean = float(self.data["Predicted"].mean())
        self.price_std = float(self.data["Predicted"].std())
        if not np.isfinite(self.price_std) or self.price_std < 1e-8:
            self.price_std = 1.0
        self.volatility_scale = float(self.data["Forecast Rolling Volatility"].quantile(0.95))
        if (
            not np.isfinite(self.volatility_scale)
            or self.volatility_scale < 1e-8
        ):
            self.volatility_scale = 1.0
        

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low = np.full(8, -10.0, dtype = np.float32),
            high = np.full(8, 10.0, dtype = np.float32),
            dtype = np.float32,
        )

        self.current_index = 0
        self.start_index = 0
        self.end_index = 0

        self.soc_mwh = self.initial_soc_mwh
        self.cumulative_profit = 0.0
    
    def _get_observation(self) -> np.ndarray:
        row = self.data.iloc[self.current_index]
        timestamp = row["time_stamp"]

        forecast_normalized = (
            float(row["Predicted"]) - self.price_mean
        ) / self.price_std

        mean_normalized = (
            float(row["Forecast Rolling Mean"]) - self.price_mean
        ) / self.price_std

        volatility_normalized = (
            float(row["Forecast Rolling Volatility"])
        ) / self.volatility_scale

        soc_normalized = (
            self.soc_mwh - self.minimum_soc_mwh
        ) / (
            self.capacity_mwh - self.minimum_soc_mwh
        )

        hour = (
            timestamp.hour + timestamp.minute / 60.0
        )

        hour_angle = 2.0 * np.pi * hour / 24.0
        weekday_angle = (
            2.0 * np.pi * timestamp.dayofweek / 7.0
        )

        observation = np.array(
            [
                forecast_normalized,
                mean_normalized,
                volatility_normalized,
                soc_normalized,
                np.sin(hour_angle),
                np.cos(hour_angle),
                np.sin(weekday_angle),
                np.cos(weekday_angle),
            ],
            dtype = np.float32,
        )

        return np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high,
        )
    
    def _feasible_charge_power(self) -> float:
        remaining_capacity_mwh = (
            self.capacity_mwh - self.soc_mwh
        )

        soc_limited_power = (
            remaining_capacity_mwh
            / (
                self.charge_efficiency
                * self.interval_hours
            )
        )

        return max(
            0.0,
            min(
                self.max_charge_mw,
                soc_limited_power,
            ),
        )
    
    def _feasible_discharge_power(self) -> float:
        available_energy_mwh = (
            self.soc_mwh - self.minimum_soc_mwh
        )

        soc_limited_power = (
            available_energy_mwh
            * self.discharge_efficiency
            / self.interval_hours
        )

        return max(
            0.0,
            min(
                self.max_discharge_mw,
                soc_limited_power,
            ),
        )

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed = seed)

        maximum_start = (
            len(self.data) - self.episode_length
        )

        if self.random_start and maximum_start > 0:
            self.start_index = int(
                self.np_random.integers(
                    0,
                    maximum_start + 1,
                )
            )
        else:
            self.start_index = 0
        
        self.current_index = self.start_index

        self.end_index = min(
            self.start_index + self.episode_length,
            len(self.data),
        )

        self.soc_mwh = self.initial_soc_mwh
        self.cumulative_profit = 0.0
        self.history = []

        observation = self._get_observation()

        info = {
            "time_stamp": self.data.loc[self.current_index, "time_stamp"],
            "SOC MWh": self.soc_mwh,
            "Cumulative Profit": self.cumulative_profit,
        }

        return observation, info
    
    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any]
    ]:
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}."
                "Expected 0, 1, or 2."
            )
        row = self.data.iloc[self.current_index]

        actual_price = float(row["Actual"])
        forecast_price = float(row["Predicted"])

        soc_start_mwh = self.soc_mwh

        charge_mw = 0.0
        discharge_mw = 0.0
        action_name = "Hold"

        if action == 0:
            charge_mw = self._feasible_charge_power()

            if charge_mw > 1e-9:
                action_name = "Charge"
        
        elif action == 2:
            discharge_mw = self._feasible_discharge_power()

            if discharge_mw > 1e-9:
                action_name = "Discharge"
        
        charged_energy_mwh = charge_mw * self.interval_hours
        discharged_energy_mwh = discharge_mw * self.interval_hours

        self.soc_mwh = (
            self.soc_mwh
            + charged_energy_mwh
            * self.charge_efficiency
            - discharged_energy_mwh
            / self.discharge_efficiency
        )

        self.soc_mwh = float(
            np.clip(
                self.soc_mwh,
                self.minimum_soc_mwh,
                self.capacity_mwh,
            )
        )

        gross_revenue = (
            (discharge_mw - charge_mw) * self.interval_hours * actual_price
        )
        throughput_mwh = (
            charge_mw + discharge_mw
        ) * self.interval_hours
        degradation_cost = (
            self.degradation_cost_per_mwh * throughput_mwh
        )
        transaction_cost = (
            self.transaction_cost_per_mwh * throughput_mwh
        )
        realized_profit = gross_revenue - degradation_cost - transaction_cost

        self.cumulative_profit += realized_profit

        self.current_index += 1

        terminated = (
            self.current_index >= self.end_index or self.current_index >= len(self.data)
        )

        truncated = False
        terminal_soc_adjustment = 0.0
        if terminated:
            terminal_reference_price = float(row["Forecast Rolling Mean"])
            terminal_soc_difference = self.soc_mwh - self.initial_soc_mwh
            terminal_soc_adjustment = terminal_soc_difference * terminal_reference_price
            observation_index = min(
                self.current_index - 1,
                len(self.data) - 1,
            )
            original_index = self.current_index
            self.current_index = observation_index
            observation = self._get_observation()
            self.current_index = original_index
        else:
            observation = self._get_observation()
        
        training_reward_value = realized_profit + terminal_soc_adjustment
        reward = float(training_reward_value / self.reward_scale)

        info = {
            "time_stamp": row["time_stamp"],
            "Forecast Price": forecast_price,
            "Actual Price": actual_price,
            "Action Name": action_name,
            "Charge MW": charge_mw,
            "Discharge MW": discharge_mw,
            "SOC Start MWh": soc_start_mwh,
            "SOC MWh": self.soc_mwh,
            "Gross Revenue": gross_revenue,
            "Degradation Cost": degradation_cost,
            "Transaction Cost": transaction_cost,
            "Realized Revenue": realized_profit,
            "Cumulative Profit": self.cumulative_profit,
        }
        self.history.append(info)
        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )
    
    def get_episode_results(self) -> pd.DataFrame:
        results = pd.DataFrame(self.history)

        if results.empty:
            return results
        
        results["Cumulative Profit"] = (
            results["Realized Revenue"].cumsum()
        )
        return results

def train_ppo_agent(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    output_directory: str = "models/ppo",
    total_timesteps: int = 250_000,
    episode_length: int = 96 * 30,
    seed: int = 42,
) -> PPO:
    """
    Train on earlier data and evaluate it on separate validation data
    """

    output_dir = Path(output_directory)
    output_dir.mkdir(parents = True, exist_ok = True)
    print("Creating training env")
    training_env = Monitor(
        BatteryTradingEnv(
            data = training_data,
            episode_length = episode_length,
            random_start = False,
        ),
        filename = str(output_dir / "validation_monitor.csv"),
    )
    print("Training env created")
    print("Creating validation env")
    validation_env = Monitor(
        BatteryTradingEnv(
            data = validation_data,
            episode_length = len(validation_data),
            random_start = False,
        ),
        filename = str(output_dir / "validation_monitor.csv"),
    )
    print("Validation env created")
    checkpoint_callback = CheckpointCallback(
        save_freq = 25_000,
        save_path = str(output_dir / "checkpoints"),
        name_prefix = "ppo_battery",
    )

    evaluation_callback = EvalCallback(
        validation_env,
        best_model_save_path = str(output_dir / "best"),
        log_path = str(output_dir / "evaluation"),
        eval_freq = 10_000,
        n_eval_episodes = 1,
        deterministic = True,
        render = False,
    )
    print("Creating PPO model")
    model = PPO(
        policy = "MlpPolicy",
        env = training_env,
        learning_rate = 3e-4,
        n_steps = 2048,
        batch_size = 64,
        n_epochs = 10,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_range = 0.2,
        ent_coef = 0.5,
        max_grad_norm = 0.5,
        policy_kwargs = {
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 64],
            }
        },
        verbose = 1,
        seed = seed,
        tensorboard_log = str(output_dir / "tensorboard"),
    )
    print("PPO model created")
    print("Preparing to learn...")
    model.learn(
        total_timesteps = total_timesteps,
        callback = [
            checkpoint_callback,
            evaluation_callback,
        ],
        progress_bar = True,
    )

    model.save(output_dir / "final_ppo_model")

    training_env.close()
    validation_env.close()
    print("Training finished")
    return model

def evaluate_ppo_agent(
    model: PPO,
    evaluation_data: pd.DataFrame,
    deterministic: bool = True,
) -> pd.DataFrame:
    print("Evaluating agent...")
    env = BatteryTradingEnv(
        data = evaluation_data,
        episode_length = len(evaluation_data),
        random_start = True,
    )

    observation, _ = env.reset(seed = 42)

    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        action, _ = model.predict(
            observation,
            deterministic = deterministic,
        )

        (
            observation,
            _,
            terminated,
            truncated,
            _,
        ) = env.step(int(action))
    
    results = env.get_episode_results()

    env.close()

    return results