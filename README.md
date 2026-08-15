# End-to-End ERCOT Battery Trading & Optimization Platform

A research platform that forecasts ERCOT real-time electricity prices with machine learning, dispatches a grid-scale battery against those forecasts using mixed-integer optimization, and measures what forecast quality and foresight are actually worth by benchmarking the result against perfect information, a rules-based baseline, and an RL agent.

This is an extension of ERCOT Electricity Price Forecasting (May–Jun 2026), which was a single-model forecasting project. Here it becomes a full trading and optimization system: larger dataset, wider model benchmark, and the complete dispatch and benchmarking stack on top.

Everything below is a simulated backtest. No live trading, no real capital. Read the Limitations section before taking any number at face value.

## Headline Results

181 out-of-sample days (Jan 1 – Jun 30, 2026) at 15-minute settlement intervals. Battery parameters and cost assumptions are identical across all five strategies.

| Strategy | Simulated profit | % of ceiling | Notes |
| --- | --- | --- | --- |
| Perfect foresight (daily MILP on actual prices) | $525,740 | 100% | Theoretical maximum |
| 24-hour day-ahead MILP (forecast-driven) | $455,139 | 87% | Full-day commitment |
| 4-hour rolling-horizon MILP (main system) | $366,059 | 70% | Re-optimizes every 15 min |
| Rules-based baseline | $157,801 | 30% | Adaptive quantile thresholds |
| PPO reinforcement learning agent | $114,853 | 22% | Custom Gymnasium env |

Three things fall out of that ladder.

The first is the value of the forecast. Under a day-ahead structure, LightGBM captures 87% of the theoretical arbitrage value, so forecast error costs roughly $71K over the period.

The second is the value of foresight. Cutting the optimizer down to a realistic 4-hour rolling horizon costs another ~$89K; the 24h run earns 24% more than the 4h run. The rolling structure is still the one that matches how a real-time desk actually operates, which is why it's the main system.

The third is the value of structure. The MILP beats the rules-based heuristic by 132%, and PPO recovers only 31% of MILP profit. When the dynamics are known and the objective is linear, structured optimization outperforms a learned policy at this training budget.

Risk profile for the main 4h rolling system: 89.5% profitable days, 1.88 interval-level profit factor, $14.3K max drawdown on interval-level equity, and about 0.64 full cycles per day.

## Architecture

```
ERCOT RTM prices + load + weather (2021-2026, 15-min)
                        │
                        ▼
        Feature engineering (54 model features)
   price lags · rolling means/vols · load dynamics
   wind/solar/temperature · cyclical time encodings
                        │
                        ▼
     Model benchmark (6 models) ──► LightGBM selected on 2025 holdout
                        │              ($23.15/MWh RMSE, $4.59 MAE)
                        ▼
              2026 price forecasts (parquet)
                        │
   ┌────────────────────┼──────────────────────────────────┐
   │                    │                                  │
   ├─► 24h day-ahead MILP dispatch (CVXPY + HiGHS, daily solves)
   ├─► 4h rolling-horizon MILP dispatch (~17,400 sequential solves)  ◄── main system
   ├─► Perfect-foresight MILP (actual prices in, ceiling benchmark)
   ├─► Rules-based dispatch baseline
   └─► PPO agent (custom Gymnasium env, Stable-Baselines3)
                        │
                        ▼
          Unified settlement & metrics
   realized P&L vs. actual prices · daily aggregation
   Sharpe · drawdown · cycles · capture ratios
                        │
                        ▼
              Plotly visualizations
```

Every strategy settles against actual prices through the same `calculate_realized_profit` path, so P&L is directly comparable up and down the ladder.

## Data

**Prices.** ERCOT Houston Hub real-time settlement point prices, 15-minute intervals, 2021–2026 (~192,000 observations), downloaded directly from ERCOT.

**Load.** ERCOT system load at matching cadence, also from ERCOT.

**Weather.** Temperature, wind, solar radiation, cloud cover, dewpoint, and precipitation for Houston, from the Open-Meteo historical weather API.

Splits are chronological with no shuffling:

- Train: 2021–2024
- Holdout test: 2025 (model selection and all reported forecast metrics)
- Backtest: Jan–Jun 2026 (never touched during model development)

## Forecasting

LightGBM was the strongest of six models on the 2025 holdout, at $23.15/MWh RMSE and $4.59 MAE. The rest of the benchmark table (tuned LightGBM, XGBoost, Random Forest, Linear Regression, tuned XGBoost) comes from the original feature set and hasn't been rerun since the fix described in Limitation 9, so I've left those numbers out rather than quote stale figures.

Diagnostics beyond headline error:

- **Monthly error profile.** RMSE ranges from about $4 (Dec) to about $49 (Apr). Error concentrates in spike-prone months while MAE stays in the $2–7 band, so the model is accurate in level and misses the extremes (see Limitation 5).
- **Permutation importance (out-of-sample).** Price 15 Minutes Ago dominates every other feature by a wide margin. The model is substantially persistence plus corrections (Limitation 4).
- **SHAP summary** on 5,000 holdout observations, for the direction and magnitude of feature effects.
- **Residual diagnostics.** Residuals vs. time, residual distribution (heavy-tailed, centered), and residual vs. actual price, which shows systematic under-prediction of spikes.

## Optimization

Battery arbitrage is formulated as a mixed-integer linear program in CVXPY and solved with HiGHS.

Parameters live in `config.py`: 100 MWh capacity, 25 MW max charge/discharge, 95% charge and 95% discharge efficiency (~90.25% round-trip), 10 MWh minimum SOC reserve, 50 MWh initial SOC, 15-minute intervals, $5.00/MWh degradation cost and $0.50/MWh transaction cost, both applied to total throughput (charge + discharge).

Decision variables per interval `t`: charge power `c_t ≥ 0`, discharge power `d_t ≥ 0`, state of charge `s_t`, and a binary mode `m_t` enforcing mutual exclusivity.

Dynamics and constraints:

```
s_{t+1} = s_t + c_t · η_c · Δt − (d_t / η_d) · Δt
s_min ≤ s_t ≤ s_max
c_t ≤ c_max · m_t
d_t ≤ d_max · (1 − m_t)
s_0 = s_initial          (daily solves also fix s_T = s_initial)
```

Objective:

```
maximize  Σ p̂_t (d_t − c_t) Δt − (κ_deg + κ_txn) Σ (c_t + d_t) Δt
```

where `p̂` is the forecast, or the actual price for the ceiling run.

Three dispatch structures:

| Structure | Solve granularity | Horizon | SOC continuity | Terminal constraint |
| --- | --- | --- | --- | --- |
| Day-ahead | 1 solve / day | 96 intervals | Resets daily | s_T = 50 MWh |
| Rolling (main) | 1 solve / interval, commit first step | 16 intervals (4h) | Continuous all year | None |
| Perfect foresight | 1 solve / day | 96 intervals | Resets daily | s_T = 50 MWh |

The rolling structure re-plans every 15 minutes using the latest SOC, which is the receding-horizon control pattern used in real-time operations. The day-ahead terminal SOC reset prevents end-of-window liquidation artifacts, at the cost of excluding cross-midnight arbitrage (Limitation 7).

## Benchmarks

### Rules-based baseline

Adaptive quantile-threshold dispatch with zero foresight. At each interval it computes the 25th and 75th percentiles of the trailing 7 days (672 intervals) of lagged forecasts, shifted one step so the thresholds never embed the current forecast. It charges at maximum feasible power when the forecast sits at or below the lower threshold, discharges at or above the upper threshold, and otherwise holds. A 96-interval minimum history gates the first day. SOC bounds, efficiencies, and costs are identical to the MILP.

It earns $157,801 over the period, which I think is a fair fight: it consumes the same forecast series, adapts its thresholds to regime shifts, and never looks ahead. What it lacks is exactly what the MILP has, which is coordinated multi-interval planning.

### PPO reinforcement learning agent

Custom Gymnasium environment (`BatteryTradingEnv`):

- **Observation (8-dim):** normalized forecast price, 24h rolling forecast mean and volatility (shifted to avoid lookahead), normalized SOC, and sine/cosine encodings of hour and weekday.
- **Actions:** discrete {charge, hold, discharge} at maximum feasible power, clamped to SOC feasibility.
- **Reward:** realized market revenue minus degradation and transaction costs, scaled. Economics are identical to the MILP objective.
- **Training:** Stable-Baselines3 PPO, MlpPolicy, 250,000 timesteps, week-long episodes with randomized starts, evaluation callback on a held-out validation slice.

Result: $114,853, or 31% of MILP profit and 73% of the rules baseline. I treat this as a deliberately vanilla floor for model-free RL on this problem rather than a ceiling (Limitation 12).

## Visualizations

Interactive Plotly output (HTML) with PNG export: data-split timeline, model comparison, actual vs. predicted with rangeslider, 4-panel residual diagnostics, split-gain / permutation / SHAP importance, monthly error profile, single-day dispatch (price, dispatch, and SOC panels), and cumulative profit with a drawdown subplot.

Planned: strategy-ladder capture chart, hour × month dispatch heatmap, profit-concentration curve, and spread capture by month.

## Repository Structure

```
├── main.py                        # End-to-end pipeline (cache flags in config.py)
├── config.py                      # All parameters: features, models, battery, rules
├── data/
│   ├── raw/                       # ERCOT + Open-Meteo downloads
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── features.py                # 54 model features (60 engineered)
│   ├── train.py                   # Model benchmark & selection
│   ├── evaluate.py
│   ├── optimization.py            # MILP: daily, rolling, perfect-foresight
│   ├── rules_based.py             # Quantile-threshold baseline
│   ├── reinforcement_learning.py  # Gymnasium env + PPO training/eval
│   ├── visualize.py               # Plotly figures (HTML + PNG)
│   └── utils.py
├── models/
│   ├── best_model.pkl             # Frozen LightGBM
│   └── ppo/                       # Checkpoints, best model, eval logs, tensorboard
└── outputs/
    ├── predictions/               # Forecast parquets
    ├── optimization/              # Dispatch parquets + metric summaries (incl. rules/PPO)
    ├── figures/
    └── cache_variables/
```

## Setup

Developed on Python 3.14.5. Install dependencies:

```bash
pip install -r requirements.txt
```

Pipeline stages are cached to parquet/JSON/pkl and controlled by the `USE_*_CACHE` flags in `config.py`. Set a stage's flag to `False` to regenerate it. Raw ERCOT price and load data is downloaded from ERCOT directly; weather comes from the Open-Meteo historical API for Houston.

**Note for Apple Silicon / Python ≥ 3.13:** PyTorch's orthogonal initialization can deadlock in BLAS during `PPO()` construction. Setting `torch.set_num_threads(1)` (or `OMP_NUM_THREADS=1`) resolves it. A Python 3.11/3.12 virtualenv is the tested-support path for the RL stack.

## Limitations & Notable Details

Ordered roughly by how much each should temper your reading of the results.

**1. All results are simulated, price-taker backtests.** There's no live execution and no market impact. A 25 MW battery is small relative to ERCOT but not negligible inside scarcity intervals, and discharging 25 MW into the exact spikes the strategy targets could soften those prices. Realized spreads would likely be tighter than simulated. All dispatch settles at the real-time settlement point price in both directions, so there's no bid/offer spread, no DAM/RTM two-settlement structure, and no bidding process. The battery is assumed to transact any quantity at the printed RT price.

**2. Forecast vintage and potential intra-window information leakage.** ⚠️ This is the most important technical caveat. Predictions are generated per timestamp from features that include recent price lags, dominated by Price 15 Minutes Ago. When the optimizer at decision time `t` plans over a window using the stored predictions for `t+1 … t+k`, each of those predictions was computed with lag features that embed actual prices after `t`, which a real operator would not have. The forecast-driven results (both the 24h and 4h runs) are therefore optimistic relative to a true operational system, which would need recursive or direct multi-step forecasts generated fresh at each decision time, with error compounding over the horizon. The perfect-foresight ceiling is unaffected, and the rules-based and PPO comparisons use the same forecast series, so the relative ladder is fairer than the absolute levels. Replacing stored one-step predictions with per-window multi-step forecasting is the highest-priority methodological upgrade.

**3. Half-year backtest window, missing ERCOT's scarcity season.** Jan–Jun 2026 excludes Jul–Sep, historically the highest-value months for storage in ERCOT. Full-year economics would probably skew higher, but seasonal robustness is untested. The "annualized" Sharpe (√365 scaling applied to 181 days of winter and spring data) should be read with that in mind.

**4. The forecaster is substantially a persistence model.** Out-of-sample permutation importance puts almost all the skill on Price 15 Minutes Ago, with every other feature a distant second. The ML layer is mostly corrections on top of a persistence core, not independent price discovery. That's normal for very short-horizon RTM forecasting, but it means skill drops off fast with horizon, which is part of why Limitation 2 matters and why the main system runs a 4-hour horizon instead of 24.

**5. Systematic spike under-prediction, and profit concentration.** Residual-vs-price diagnostics show the model under-calls extremes, which is where battery money is made. Daily P&L is correspondingly fat-tailed: the best single day contributes roughly 10% of period profit, and cumulative equity is a staircase driven by a handful of scarcity events. Three consequences follow. Mean-based metrics like daily Sharpe are flattering and fragile for this distribution. Missing one or two major events (an outage, an offline battery) would materially change the period result. And the perfect-foresight gap is dominated by spike days.

**6. Metric definitions are non-standard where flagged.** Profit factor (interval-level) is gross discharge revenue divided by gross charging-plus-friction cost, so it's a round-trip spread multiple, not a trading win/loss PF; charging intervals are structurally "losses." Maximum drawdown (interval-level) measures peak intraday cash outlay while charging ahead of discharge, roughly the cost of filling the battery, not risk of realized loss. On daily P&L, perfect foresight's drawdown is $0 by construction, since every day is ≥ $0 given that zero dispatch is always feasible under the daily reset. The results reproduce that property correctly: the worst perfect-foresight day was +$1.76.

**7. Structural choices constrain the strategies.** Daily solves enforce a midnight terminal-SOC reset, which rules out cross-day arbitrage (evening peak → overnight trough → morning peak spreads) and makes the day boundary arbitrary. The rolling solver has no terminal-SOC value, so it shows end-of-window myopia: it holds far less inventory (average SOC 22 MWh vs. 59 MWh for day-ahead) and forgoes trades whose payoff sits beyond 4 hours. A terminal SOC value term is a known mitigation, left out deliberately to keep the horizon comparison clean. The 24h-vs-4h comparison isolates horizon while holding forecast vintage constant (Limitation 2). A real rolling operator would have fresher, better near-term forecasts than a day-ahead committer, which would partially offset the measured 24% gap, so the comparison measures pure horizon value rather than total operational difference.

**8. Battery physics are simplified.** Constant charge/discharge efficiencies (no SOC- or power-dependence), linear degradation cost per MWh throughput (no cycle-depth or calendar aging), no temperature derating, no auxiliary/HVAC load, no forced outages or maintenance windows, and no interconnection limits or charging demand charges. The binary charge/discharge exclusivity is enforced explicitly, though at 90.25% round-trip efficiency plus throughput costs, simultaneous charge and discharge is never profitable anyway. The backtests confirm 0 simultaneous intervals, which means the binary (and with it the MILP-vs-LP distinction) is probably removable for a large speedup.

**9. Model selection wasn't rerun.** The six-model benchmark, including the "default LightGBM beat tuned LightGBM, tuned XGBoost was worst" comparison, comes from the original feature set, before the fix that regenerated the predictions above. Only the winning LightGBM model was rerun for this update, so I haven't reconfirmed whether the same model still wins under the corrected features. Worth doing before leaning on it further. Separately, model choice rests on a single chronological holdout year (2025). There's no walk-forward re-fitting across the backtest, since the model is frozen at end-2024 training and never sees 2025–2026 data. That's conservative in one sense, but it leaves 18 months of regime drift unmodeled.

**10. Energy-only revenue understates real BESS economics.** No ancillary services (RRS, ECRS, regulation), which are a large share of real ERCOT battery revenue, no capacity payments, and no co-optimization of energy against AS. This measures the energy-arbitrage slice in isolation, so actual BESS economics would look different, and probably better, once AS revenue is layered in.

**11. Single node, no basis or congestion.** Houston Hub settlement only. A physical asset settles at its resource node, and nodal basis, congestion, and ORDC adders relative to hub prices are out of scope.

**12. The PPO result is a floor for RL, not a verdict.** The configuration is deliberately vanilla: 500k timesteps, discrete 3-action full-power actions (no partial charge rates, which the MILP uses freely), an 8-feature observation, no reward shaping, default PPO hyperparameters, and normalization statistics computed from each environment's own dataset. That last one is a mild leakage and a train/eval distribution mismatch; computing stats from training data and passing them to all envs is the planned fix. The 31%-of-MILP result quantifies the cost of ignoring known structure, not the limit of RL on this problem. Known-good extensions: continuous actions, richer observations such as forecast windows, longer training, and reward shaping toward spread capture.

**13. Reproducibility notes.** PPO training is seeded (42), but PyTorch determinism across platforms isn't guaranteed; HiGHS solves are deterministic. Optimization and visualization stages are cached to parquet/JSON via flags in `main.py`, and cached artifacts from earlier code versions can carry schema drift (for example a since-fixed misspelled revenue column), so regenerate caches after pulling changes. Plotly PNG export via kaleido mangles any title or annotation containing two `$` characters, thanks to a MathJax delimiter collision. The HTML outputs are unaffected.

## Roadmap

- Per-window multi-step forecasting to close the forecast-vintage gap (Limitation 2)
- Full-year 2026 backtest including summer scarcity (Limitation 3)
- Rerun the full model benchmark under the corrected features (Limitation 9)
- Strategy-ladder capture chart, hour × month dispatch heatmap, profit-concentration curve
- Terminal-SOC value term for the rolling solver, plus an LP relaxation speedup (Limitation 8)
- Walk-forward model re-fitting, and probabilistic/quantile forecasts for risk-aware dispatch
- Ancillary-service co-optimization (Limitation 10)
- PPO upgrades: continuous actions, shared normalization stats, longer training (Limitation 12)

## License

MIT
