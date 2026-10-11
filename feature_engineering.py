"""
feature_engineering.py
----------------------
Feature Engineering and Temporal Sequence Windowing for Wind Turbine Yaw Prediction.
Transforms raw telemetry into feature matrices with:
- Trigonometric / Cyclical encodings (sin/cos) for directional continuity
- Angular velocity and yaw rotation acceleration
- Rolling temporal statistics (turbulence intensity, moving averages)
- Non-leaking scaling (fitted strictly on train partition)
- Sliding window sequence formatting for Deep Learning models.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class YawFeaturePipeline:
    def __init__(self, sequence_length: int = 30, forecast_horizon: int = 1):
        """
        Parameters:
            sequence_length (int): Lookback history window length in time steps.
            forecast_horizon (int): Number of steps ahead to predict (1 = next step).
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []

    def construct_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers physics-informed and cyclical features from SCADA telemetry.
        """
        data = df.copy()
        
        # 1. Cyclical Angular Encoding (sin / cos decomposition)
        # Avoids 0 deg <-> 360 deg boundary jump discontinuity
        for col in ["wind_direction", "nacelle_yaw_angle"]:
            rad = np.radians(data[col])
            data[f"{col}_sin"] = np.sin(rad)
            data[f"{col}_cos"] = np.cos(rad)

        # Yaw misalignment cyclical encoding
        misalign_rad = np.radians(data["yaw_misalignment"])
        data["yaw_misalignment_sin"] = np.sin(misalign_rad)
        data["yaw_misalignment_cos"] = np.cos(misalign_rad)

        # 2. Dynamic Yaw Rotational Metrics & Derivatives
        # Angular change between successive time steps wrapped in [-180, 180]
        yaw_diff = (data["nacelle_yaw_angle"].diff() + 180.0) % 360.0 - 180.0
        data["yaw_rate_deg_min"] = yaw_diff.fillna(0.0)
        
        # Yaw acceleration
        data["yaw_accel"] = data["yaw_rate_deg_min"].diff().fillna(0.0)

        # Wind speed rate of change (gust intensity indicator)
        data["wind_speed_delta"] = data["wind_speed"].diff().fillna(0.0)

        # 3. Rolling Temporal Aggregations (Turbulence & Inertia)
        rolling_windows = [5, 15]
        for w in rolling_windows:
            data[f"wind_speed_roll_mean_{w}"] = data["wind_speed"].rolling(window=w, min_periods=1).mean()
            data[f"wind_speed_roll_std_{w}"] = data["wind_speed"].rolling(window=w, min_periods=1).std().fillna(0.0)
            data[f"yaw_misalign_roll_std_{w}"] = data["yaw_misalignment"].rolling(window=w, min_periods=1).std().fillna(0.0)

        # 4. Aerodynamic Power Proxy (Effective wind power accounting for misalignment)
        data["effective_wind_power_proxy"] = (data["wind_speed"] ** 3) * (np.cos(misalign_rad) ** 2)

        return data

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Dict[str, Any]:
        """
        Builds sliding window sequences and splits into temporal Train / Val / Test sets.
        
        Returns:
            Dict containing X_train, y_train, X_val, y_val, X_test, y_test,
            along with original unscaled yaw angles for inverse evaluation.
        """
        featured_df = self.construct_features(df)
        
        # Define feature subset (excluding target, raw timestamp, and high-level raw angle)
        exclude_cols = ["timestamp", "nacelle_yaw_angle"]
        self.feature_columns = [c for c in featured_df.columns if c not in exclude_cols]

        X_raw = featured_df[self.feature_columns].values
        # Target: Nacelle yaw angle at step t + forecast_horizon (in degrees [0, 360))
        y_raw = featured_df["nacelle_yaw_angle"].values

        n_samples = len(featured_df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        # Fit scaler ONLY on training data to strictly prevent data leakage
        self.scaler.fit(X_raw[:train_end])
        X_scaled = self.scaler.transform(X_raw)

        # Generate sliding windows
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.sequence_length - self.forecast_horizon + 1):
            X_seq.append(X_scaled[i : i + self.sequence_length])
            # Target is the yaw angle at the horizon step
            target_idx = i + self.sequence_length + self.forecast_horizon - 1
            y_seq.append(y_raw[target_idx])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        # Split sequence indices
        seq_train_end = train_end - self.sequence_length
        seq_val_end = val_end - self.sequence_length

        data_bundle = {
            "X_train": X_seq[:seq_train_end],
            "y_train": y_seq[:seq_train_end],
            "X_val": X_seq[seq_train_end:seq_val_end],
            "y_val": y_seq[seq_train_end:seq_val_end],
            "X_test": X_seq[seq_val_end:],
            "y_test": y_seq[seq_val_end:],
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
            "sequence_length": self.sequence_length
        }

        print("=" * 70)
        print("TEMPORAL SEQUENCE PARTITIONING")
        print("=" * 70)
        print(f"Sequence Length (Lookback) : {self.sequence_length} steps")
        print(f"Number of Features         : {len(self.feature_columns)}")
        print(f"Training Tensor Shape      : X={data_bundle['X_train'].shape}, y={data_bundle['y_train'].shape}")
        print(f"Validation Tensor Shape    : X={data_bundle['X_val'].shape}, y={data_bundle['y_val'].shape}")
        print(f"Testing Tensor Shape       : X={data_bundle['X_test'].shape}, y={data_bundle['y_test'].shape}")
        print("-" * 70)

        return data_bundle


if __name__ == "__main__":
    from scada_simulator import generate_scada_yaw_dataset
    df = generate_scada_yaw_dataset(n_samples=1500)
    pipeline = YawFeaturePipeline(sequence_length=20)
    bundle = pipeline.prepare_sequences(df)
    print("Feature Pipeline test completed successfully.")
