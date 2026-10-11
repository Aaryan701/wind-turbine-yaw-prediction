"""
eda_visualizer.py
-----------------
Comprehensive Exploratory Data Analysis (EDA) and Preprocessing for
Wind Turbine SCADA Yaw & Telemetry Time-Series.
Includes:
- Missing value audit and angular-aware interpolation
- Temporal continuity verification
- Seaborn & Matplotlib visualizer suite:
    1. Time-series tracking (Wind Direction vs. Nacelle Yaw vs. Misalignment)
    2. Circular / Polar Rose distribution
    3. Feature distributions & Kernel Density Estimation (KDE)
    4. Correlation matrix heatmap
    5. Yaw error vs. Active Power aerodynamic degradation curve
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class WindTurbineEDA:
    def __init__(self, df: pd.DataFrame, output_dir: str = "eda_plots"):
        """
        Initializes EDA tool with SCADA DataFrame.
        """
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 10

    def inspect_and_clean(self) -> pd.DataFrame:
        """
        Audits data completeness, cleans anomalies, and performs angular-safe interpolation.
        
        Angular interpolation rationale:
        Naively interpolating between 359 deg and 1 deg yields 180 deg (opposite direction!).
        Correct approach: decompose theta into (sin theta, cos theta), interpolate Cartesian vectors,
        and reconstruct theta via arctan2(y, x).
        """
        print("=" * 70)
        print("SCADA TELEMETRY AUDIT & DATA CLEANING")
        print("=" * 70)
        
        # 1. Temporal Index Verification
        if not pd.api.types.is_datetime64_any_dtype(self.df["timestamp"]):
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)
        
        # Summary statistics
        print(f"Total Observations : {len(self.df):,}")
        print(f"Time Range         : {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"Missing Values Summary:\n{self.df.isnull().sum()[self.df.isnull().sum() > 0]}\n")
        
        # 2. Angular-Safe Interpolation for Circular Features
        angular_cols = ["wind_direction", "nacelle_yaw_angle"]
        for col in angular_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                rad = np.radians(self.df[col])
                sin_comp = np.sin(rad).interpolate(method="time" if "timestamp" in self.df else "linear")
                cos_comp = np.cos(rad).interpolate(method="time" if "timestamp" in self.df else "linear")
                # Reconstruct angles into [0, 360)
                reconstructed_deg = np.degrees(np.arctan2(sin_comp, cos_comp)) % 360.0
                self.df[col] = reconstructed_deg
        
        # 3. Standard Continuous Feature Interpolation
        continuous_cols = ["wind_speed", "rotor_speed", "active_power", "ambient_temperature"]
        for col in continuous_cols:
            if col in self.df.columns and self.df[col].isnull().any():
                self.df[col] = self.df[col].interpolate(method="linear").bfill().ffill()

        # 4. Recompute Yaw Misalignment rigorously in [-180, 180] deg
        self.df["yaw_misalignment"] = (
            (self.df["wind_direction"] - self.df["nacelle_yaw_angle"] + 180.0) % 360.0 - 180.0
        )
        
        print(f"Post-cleaning Missing Values Total: {self.df.isnull().sum().sum()}")
        print("-" * 70)
        return self.df

    def plot_time_series_tracking(self, window_steps: int = 300, filename: str = "1_time_series_tracking.png"):
        """
        Visualizes Wind Direction vs Nacelle Yaw Angle tracking performance and yaw error.
        """
        subset = self.df.iloc[:window_steps]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})
        
        # Subplot 1: Angular Tracking
        ax1.plot(subset["timestamp"], subset["wind_direction"], label="Wind Direction (deg)", color="#1f77b4", lw=1.5, alpha=0.85)
        ax1.plot(subset["timestamp"], subset["nacelle_yaw_angle"], label="Nacelle Yaw Angle (deg)", color="#ff7f0e", lw=2.0)
        ax1.set_ylabel("Angle (degrees)", fontsize=11, fontweight="bold")
        ax1.set_title("Wind Turbine Yaw Tracking Dynamics & Nacelle Deadband Response", fontsize=13, fontweight="bold", pad=12)
        ax1.legend(loc="upper right", frameon=True)
        ax1.set_ylim(-5, 365)
        
        # Subplot 2: Yaw Misalignment Error
        ax2.plot(subset["timestamp"], subset["yaw_misalignment"], label="Yaw Misalignment Error (deg)", color="#d62728", lw=1.2)
        ax2.axhline(0, color="black", linestyle="--", alpha=0.5, lw=1)
        ax2.axhline(7.0, color="gray", linestyle=":", label="Deadband Threshold (+/-7°)", lw=1)
        ax2.axhline(-7.0, color="gray", linestyle=":", lw=1)
        ax2.fill_between(subset["timestamp"], subset["yaw_misalignment"], 0, alpha=0.2, color="#d62728")
        ax2.set_ylabel("Misalignment (deg)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Timestamp", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", frameon=True)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA Export] Saved: {save_path}")

    def plot_polar_wind_rose(self, filename: str = "2_polar_wind_rose.png"):
        """
        Polar wind rose comparing distributions of wind direction and nacelle orientation.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"})
        
        bins = np.linspace(0, 2 * np.pi, 37)
        # Wind direction histogram in polar coords
        hist_wd, _ = np.histogram(np.radians(self.df["wind_direction"]), bins=bins)
        # Nacelle yaw histogram
        hist_yaw, _ = np.histogram(np.radians(self.df["nacelle_yaw_angle"]), bins=bins)
        
        width = 2 * np.pi / 36
        centers = bins[:-1] + width / 2
        
        # Wind Direction
        axes[0].bar(centers, hist_wd, width=width, bottom=0.0, color="#1f77b4", alpha=0.7, edgecolor="black")
        axes[0].set_theta_zero_location("N")
        axes[0].set_theta_direction(-1)  # Clockwise
        axes[0].set_title("Wind Direction Distribution", fontsize=12, fontweight="bold", pad=15)
        
        # Nacelle Yaw
        axes[1].bar(centers, hist_yaw, width=width, bottom=0.0, color="#ff7f0e", alpha=0.7, edgecolor="black")
        axes[1].set_theta_zero_location("N")
        axes[1].set_theta_direction(-1)
        axes[1].set_title("Nacelle Yaw Angle Distribution", fontsize=12, fontweight="bold", pad=15)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA Export] Saved: {save_path}")

    def plot_distributions_and_kde(self, filename: str = "3_feature_distributions.png"):
        """
        Visualizes density distributions of core SCADA metrics.
        """
        features = ["wind_speed", "yaw_misalignment", "rotor_speed", "active_power"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        
        colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for idx, col in enumerate(features):
            sns.histplot(self.df[col], kde=True, ax=axes[idx], color=colors[idx], stat="density", bins=35)
            axes[idx].set_title(f"Distribution: {col.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Density")
            
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA Export] Saved: {save_path}")

    def plot_correlation_heatmap(self, filename: str = "4_correlation_heatmap.png"):
        """
        Computes Pearson correlation matrix across telemetry features.
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr(method="pearson")
        
        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("SCADA Telemetry Pearson Correlation Matrix", fontsize=12, fontweight="bold", pad=12)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA Export] Saved: {save_path}")

    def plot_yaw_misalignment_vs_power(self, filename: str = "5_yaw_error_vs_power.png"):
        """
        Illustrates relationship between yaw misalignment and power output (aerodynamic cosine loss).
        """
        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(
            self.df["yaw_misalignment"], self.df["active_power"],
            c=self.df["wind_speed"], cmap="viridis", alpha=0.6, s=18, edgecolors="none"
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Wind Speed (m/s)", fontsize=10, fontweight="bold")
        
        ax.set_xlabel("Yaw Misalignment Error (deg)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Active Power Output (kW)", fontsize=11, fontweight="bold")
        ax.set_title("Impact of Yaw Misalignment Angle on Aerodynamic Power Generation", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(-40, 40)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[EDA Export] Saved: {save_path}")


if __name__ == "__main__":
    from scada_simulator import generate_scada_yaw_dataset
    raw_df = generate_scada_yaw_dataset(n_samples=2000)
    eda = WindTurbineEDA(raw_df)
    clean_df = eda.inspect_and_clean()
    eda.plot_time_series_tracking()
    eda.plot_polar_wind_rose()
    eda.plot_distributions_and_kde()
    eda.plot_correlation_heatmap()
    eda.plot_yaw_misalignment_vs_power()
