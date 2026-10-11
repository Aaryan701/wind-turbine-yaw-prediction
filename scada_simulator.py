"""
scada_simulator.py
------------------
Realistic Wind Turbine SCADA Time-Series Data Generator.
Simulates aerodynamic, rotational, and yaw control dynamics including:
- Wind speed (Ornstein-Uhlenbeck process + turbulent gusts)
- Wind direction (drift + stochastic fluctuations in [0, 360) deg)
- Nacelle yaw angle with deadband hysteresis control (yaw drive mechanism)
- Rotor speed, active power (cubic power curve with cos^p(misalignment) loss)
- Synthetic missing values and sensor noise for preprocessing demonstration.
"""

import numpy as np
import pandas as pd


def generate_scada_yaw_dataset(
    n_samples: int = 5000,
    sampling_interval_min: int = 1,
    random_state: int = 42,
    missing_rate: float = 0.02
) -> pd.DataFrame:
    """
    Generates realistic 1-minute SCADA telemetry for a 2.0 MW wind turbine.
    
    Parameters:
        n_samples (int): Number of consecutive temporal steps.
        sampling_interval_min (int): Sampling resolution in minutes.
        random_state (int): Seed for reproducibility.
        missing_rate (float): Fraction of random missing observations to inject.
        
    Returns:
        pd.DataFrame: SCADA dataset with clean and raw telemetry.
    """
    np.random.seed(random_state)
    
    # 1. Temporal Index
    start_time = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=n_samples, freq=f"{sampling_interval_min}min")
    
    # 2. Wind Speed (Ornstein-Uhlenbeck mean-reverting stochastic process)
    # Mean wind speed ~ 8.5 m/s, reversion speed theta, volatility sigma
    mean_ws = 8.5
    theta_ws = 0.05
    sigma_ws = 0.8
    wind_speed = np.zeros(n_samples)
    wind_speed[0] = mean_ws
    
    for t in range(1, n_samples):
        # Mean reversion + Gaussian gust perturbation
        dw = np.random.normal(0, 1)
        wind_speed[t] = wind_speed[t - 1] + theta_ws * (mean_ws - wind_speed[t - 1]) + sigma_ws * dw
    wind_speed = np.clip(wind_speed, 0.5, 25.0)  # Cut-in to cut-out envelope
    
    # 3. Wind Direction (Cyclical drift + stochastic turbulence in [0, 360))
    wind_dir = np.zeros(n_samples)
    wind_dir[0] = 180.0  # Initial South-facing wind
    for t in range(1, n_samples):
        # Low frequency diurnal drift + stochastic turbulence
        drift = 0.5 * np.sin(2 * np.pi * t / (1440 / sampling_interval_min))
        gust_dir = np.random.normal(0, 2.2)
        wind_dir[t] = (wind_dir[t - 1] + drift + gust_dir) % 360.0

    # 4. Nacelle Yaw Angle with Physical Deadband Controller
    # Real wind turbines do not yaw continuously to prevent wear on yaw bearings/motors.
    # Hysteresis rule: Yaw only triggers when |misalignment| > deadband_threshold for > N steps
    deadband_threshold_deg = 7.0  # Common industry deadband (+/- 7 degrees)
    max_yaw_rate = 0.5 * (sampling_interval_min * 60)  # 0.5 deg/s max motor rotational speed
    
    nacelle_yaw = np.zeros(n_samples)
    nacelle_yaw[0] = wind_dir[0]  # Initially aligned
    
    yaw_error_accum = 0.0
    for t in range(1, n_samples):
        current_yaw = nacelle_yaw[t - 1]
        target_dir = wind_dir[t]
        
        # Calculate shortest angular difference: range [-180, 180]
        diff = (target_dir - current_yaw + 180.0) % 360.0 - 180.0
        
        # Check deadband trigger condition
        if abs(diff) > deadband_threshold_deg:
            # Active yawing towards the wind
            step_yaw = np.sign(diff) * min(abs(diff), max_yaw_rate)
            nacelle_yaw[t] = (current_yaw + step_yaw) % 360.0
        else:
            # Stationary nacelle (within acceptable deadband window)
            nacelle_yaw[t] = current_yaw

    # 5. Yaw Misalignment (Relative Error angle in [-180, 180] deg)
    yaw_misalignment = (wind_dir - nacelle_yaw + 180.0) % 360.0 - 180.0

    # 6. Rotor Speed (RPM) and Active Power Output (kW)
    # Non-linear cubic power equation with yaw misalignment cosine loss
    rotor_speed = np.clip(1.2 * wind_speed + np.random.normal(0, 0.3, n_samples), 6.0, 18.0)
    
    # Power loss due to yaw misalignment is traditionally modelled as P = P_0 * cos^p(misalignment), p ~ 2 to 3
    aerodynamic_loss_factor = np.cos(np.radians(yaw_misalignment)) ** 2
    raw_power = 0.5 * 1.225 * (np.pi * (45.0 ** 2)) * (wind_speed ** 3) * 0.45 / 1000.0  # kW
    active_power = np.clip(raw_power * aerodynamic_loss_factor, 0.0, 2000.0)  # 2.0 MW rating
    
    # 7. Ambient Temperature (deg C)
    ambient_temp = 15.0 + 5.0 * np.sin(2 * np.pi * np.arange(n_samples) / (1440 / sampling_interval_min)) + np.random.normal(0, 0.5, n_samples)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "wind_speed": wind_speed,
        "wind_direction": wind_dir,
        "nacelle_yaw_angle": nacelle_yaw,
        "yaw_misalignment": yaw_misalignment,
        "rotor_speed": rotor_speed,
        "active_power": active_power,
        "ambient_temperature": ambient_temp
    })

    # 8. Inject Realistic Sensor Dropouts / Missing Values
    if missing_rate > 0:
        mask = np.random.rand(n_samples) < missing_rate
        cols_to_corrupt = ["wind_speed", "wind_direction", "nacelle_yaw_angle", "active_power"]
        for col in cols_to_corrupt:
            col_mask = mask & (np.random.rand(n_samples) < 0.6)
            df.loc[col_mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_scada_yaw_dataset(n_samples=1000)
    print("Simulated SCADA Telemetry Preview:")
    print(df.head())
    print("\nMissing Values Count:")
    print(df.isnull().sum())
