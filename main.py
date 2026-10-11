"""
main.py
-------
End-to-End Orchestrator for Wind Turbine Yaw Prediction System.
Executes the full pipeline:
1. SCADA Simulation / Data Ingestion
2. Exploratory Data Analysis & Visualizations (Seaborn / Matplotlib)
3. Feature Engineering & Temporal Sequence Preparation
4. Baseline Deep Learning Model Training (Default Hyperparameters)
5. Floral Metaheuristic Optimization (Flower Pollination Algorithm)
6. Optimized Deep Learning Model Training
7. Rigorous Head-to-Head Evaluation & Benchmarking (Baseline vs FPA-Optimized)
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset

# Local Modules
from scada_simulator import generate_scada_yaw_dataset
from eda_visualizer import WindTurbineEDA
from feature_engineering import YawFeaturePipeline
from models import TemporalConvGRU, CircularMSELoss, CircularMAELoss, compute_angular_metrics
from floral_optimizer import FlowerPollinationOptimizer


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_data: tuple,
    epochs: int = 15,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    verbose_prefix: str = "Training"
) -> Tuple[torch.nn.Module, list, list, float]:
    """
    Standard training loop with Circular MSE Loss and learning rate scheduling.
    """
    criterion = CircularMSELoss()
    val_criterion = CircularMAELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    
    X_val, y_val = val_data
    X_val, y_val = X_val.to(device), y_val.to(device)
    
    train_losses = []
    val_maes = []
    
    start_time = time.time()
    best_loss = float("inf")
    best_weights = None
    
    print(f"\n--- {verbose_prefix} (Epochs: {epochs}, LR: {learning_rate:.5f}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_losses.append(loss.item())
            
        epoch_train_loss = np.mean(batch_losses)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_preds = model(X_val)
            v_mae = val_criterion(v_preds, y_val).item()
            val_maes.append(v_mae)
            
        scheduler.step(v_mae)
        
        if v_mae < best_loss:
            best_loss = v_mae
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train Circ MSE: {epoch_train_loss:.2f}°² | Val Circ MAE: {v_mae:.3f}°")
            
    elapsed_time = time.time() - start_time
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, train_losses, val_maes, elapsed_time


def main():
    print("=" * 75)
    print("      DEEP LEARNING WIND TURBINE YAW ROTATION PREDICTION SYSTEM      ")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}\n")

    # -------------------------------------------------------------------------
    # STEP 1: Data Ingestion & Preprocessing
    # -------------------------------------------------------------------------
    print(">> STEP 1: Generating & Ingesting SCADA Telemetry...")
    raw_df = generate_scada_yaw_dataset(
        n_samples=4000,
        sampling_interval_min=1,
        random_state=42,
        missing_rate=0.015
    )
    
    eda = WindTurbineEDA(raw_df, output_dir="eda_plots")
    clean_df = eda.inspect_and_clean()
    
    print("\nGenerating Exploratory Data Visualizations...")
    eda.plot_time_series_tracking(window_steps=300)
    eda.plot_polar_wind_rose()
    eda.plot_distributions_and_kde()
    eda.plot_correlation_heatmap()
    eda.plot_yaw_misalignment_vs_power()
    print("EDA visual suite exported to 'eda_plots/' directory.")

    # -------------------------------------------------------------------------
    # STEP 2: Feature Engineering & Temporal Sequence Windowing
    # -------------------------------------------------------------------------
    print("\n>> STEP 2: Feature Engineering & Sliding Window Sequence Construction...")
    pipeline = YawFeaturePipeline(sequence_length=24, forecast_horizon=1)
    bundle = pipeline.prepare_sequences(clean_df, train_ratio=0.70, val_ratio=0.15)
    
    # Convert arrays to PyTorch Tensors
    X_train_t = torch.tensor(bundle["X_train"], dtype=torch.float32)
    y_train_t = torch.tensor(bundle["y_train"], dtype=torch.float32)
    X_val_t = torch.tensor(bundle["X_val"], dtype=torch.float32)
    y_val_t = torch.tensor(bundle["y_val"], dtype=torch.float32)
    X_test_t = torch.tensor(bundle["X_test"], dtype=torch.float32)
    y_test_t = torch.tensor(bundle["y_test"], dtype=torch.float32)

    input_dim = X_train_t.shape[2]
    val_data_tuple = (X_val_t, y_val_t)

    # -------------------------------------------------------------------------
    # STEP 3: Baseline Model Training (Default Un-tuned Architecture)
    # -------------------------------------------------------------------------
    print("\n>> STEP 3: Training Baseline Deep Learning Model...")
    baseline_batch_size = 32
    baseline_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=baseline_batch_size,
        shuffle=True
    )
    
    baseline_model = TemporalConvGRU(
        input_dim=input_dim,
        conv_channels=16,
        gru_hidden_dim=32,
        gru_layers=1,
        dropout=0.1
    ).to(device)
    
    baseline_model, b_train_loss, b_val_mae, b_time = train_model(
        model=baseline_model,
        train_loader=baseline_loader,
        val_data=val_data_tuple,
        epochs=12,
        learning_rate=0.005,
        device=device,
        verbose_prefix="Baseline Model Training"
    )

    # -------------------------------------------------------------------------
    # STEP 4: Flower Pollination Algorithm (Floral Metaheuristic Tuning)
    # -------------------------------------------------------------------------
    print("\n>> STEP 4: Metaheuristic Optimization via Flower Pollination Algorithm...")
    fpa = FlowerPollinationOptimizer(
        n_flowers=6,
        max_generations=4,
        switch_prob=0.8,
        random_state=42
    )
    
    best_hyperparams, fpa_history = fpa.optimize(
        train_data=(X_train_t, y_train_t),
        val_data=(X_val_t, y_val_t),
        proxy_epochs=3,
        batch_size=32,
        device=device
    )

    # -------------------------------------------------------------------------
    # STEP 5: Optimized Model Training with Floral Hyperparameters
    # -------------------------------------------------------------------------
    print("\n>> STEP 5: Training FPA-Optimized Deep Learning Model...")
    opt_model = TemporalConvGRU(
        input_dim=input_dim,
        conv_channels=best_hyperparams["conv_channels"],
        gru_hidden_dim=best_hyperparams["gru_hidden_dim"],
        gru_layers=best_hyperparams["gru_layers"],
        dropout=best_hyperparams["dropout"]
    ).to(device)
    
    opt_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=32,
        shuffle=True
    )
    
    opt_model, opt_train_loss, opt_val_mae, opt_time = train_model(
        model=opt_model,
        train_loader=opt_loader,
        val_data=val_data_tuple,
        epochs=12,
        learning_rate=best_hyperparams["learning_rate"],
        device=device,
        verbose_prefix="FPA-Optimized Model Training"
    )

    # -------------------------------------------------------------------------
    # STEP 6: Head-to-Head Evaluation on Held-Out Test Set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(">> STEP 6: Rigorous Out-of-Sample Test Evaluation")
    print("=" * 75)
    
    baseline_model.eval()
    opt_model.eval()
    
    with torch.no_grad():
        b_test_preds = baseline_model(X_test_t.to(device)).cpu().numpy()
        opt_test_preds = opt_model(X_test_t.to(device)).cpu().numpy()
        
    y_test_np = y_test_t.numpy()
    
    b_metrics = compute_angular_metrics(y_test_np, b_test_preds)
    opt_metrics = compute_angular_metrics(y_test_np, opt_test_preds)
    
    # Calculate Improvements
    mae_impr = ((b_metrics["Circular_MAE_deg"] - opt_metrics["Circular_MAE_deg"]) / b_metrics["Circular_MAE_deg"]) * 100.0
    rmse_impr = ((b_metrics["Circular_RMSE_deg"] - opt_metrics["Circular_RMSE_deg"]) / b_metrics["Circular_RMSE_deg"]) * 100.0
    
    # Display Formatted Benchmark Comparison Table
    results_df = pd.DataFrame([
        {
            "Pipeline Configuration": "Baseline ConvGRU",
            "Circular MAE (deg)": f"{b_metrics['Circular_MAE_deg']:.3f}°",
            "Circular RMSE (deg)": f"{b_metrics['Circular_RMSE_deg']:.3f}°",
            "Cosine Alignment": f"{b_metrics['Cosine_Alignment']:.4f}",
            "R² Score": f"{b_metrics['R2_Score']:.4f}",
            "Train Time (s)": f"{b_time:.2f}s"
        },
        {
            "Pipeline Configuration": "FPA-Optimized ConvGRU",
            "Circular MAE (deg)": f"{opt_metrics['Circular_MAE_deg']:.3f}°",
            "Circular RMSE (deg)": f"{opt_metrics['Circular_RMSE_deg']:.3f}°",
            "Cosine Alignment": f"{opt_metrics['Cosine_Alignment']:.4f}",
            "R² Score": f"{opt_metrics['R2_Score']:.4f}",
            "Train Time (s)": f"{opt_time:.2f}s"
        }
    ])
    
    print("\nBENCHMARK COMPARISON MATRIX:")
    print(results_df.to_string(index=False))
    print(f"\nOptimization Gain -> Circular MAE Error Reduction  : {mae_impr:+.2f}%")
    print(f"Optimization Gain -> Circular RMSE Error Reduction : {rmse_impr:+.2f}%")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # STEP 7: Export Comparison & Prediction Visualizations
    # -------------------------------------------------------------------------
    print("\n>> STEP 7: Generating Performance Comparison Plots...")
    os.makedirs("results_plots", exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    
    # Plot A: Yaw Angle Tracking on Test Set (first 180 test points)
    eval_window = min(180, len(y_test_np))
    x_axis = np.arange(eval_window)
    
    axes[0].plot(x_axis, y_test_np[:eval_window] % 360, label="Ground Truth Nacelle Yaw (°)", color="black", lw=2.0)
    axes[0].plot(x_axis, b_test_preds[:eval_window] % 360, label="Baseline Model Prediction (°)", color="#d62728", lw=1.5, linestyle="--", alpha=0.85)
    axes[0].plot(x_axis, opt_test_preds[:eval_window] % 360, label="FPA-Optimized Prediction (°)", color="#2ca02c", lw=1.8)
    axes[0].set_title("Out-of-Sample Yaw Angle Tracking Comparison (Test Set Horizon = 1)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Yaw Angle (degrees)", fontsize=10, fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].set_ylim(-5, 365)
    
    # Plot B: Absolute Angular Error Residuals
    b_err = np.abs((b_test_preds[:eval_window] - y_test_np[:eval_window] + 180.0) % 360.0 - 180.0)
    opt_err = np.abs((opt_test_preds[:eval_window] - y_test_np[:eval_window] + 180.0) % 360.0 - 180.0)
    
    axes[1].plot(x_axis, b_err, label=f"Baseline Error (Mean: {np.mean(b_err):.2f}°)", color="#d62728", alpha=0.7)
    axes[1].plot(x_axis, opt_err, label=f"FPA-Optimized Error (Mean: {np.mean(opt_err):.2f}°)", color="#2ca02c", alpha=0.8)
    axes[1].axhline(np.mean(b_err), color="#d62728", linestyle=":", lw=1.2)
    axes[1].axhline(np.mean(opt_err), color="#2ca02c", linestyle=":", lw=1.2)
    axes[1].set_title("Instantaneous Geodesic Angular Error (|θ_pred - θ_true|)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Angular Error (°)", fontsize=10, fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)
    
    # Plot C: Floral Optimization Convergence History
    axes[2].plot(range(len(fpa_history)), fpa_history, marker="o", color="#9467bd", lw=2.0)
    axes[2].set_title("Flower Pollination Algorithm (FPA) Metaheuristic Convergence Curve", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("FPA Generation / Iteration", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Global Best Val Circ MAE (°)", fontsize=10, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    comp_plot_path = os.path.join("results_plots", "model_comparison_and_convergence.png")
    fig.savefig(comp_plot_path, dpi=300)
    plt.close(fig)
    print(f"Comparison plot exported to: {comp_plot_path}")
    print("\nEnd-to-End Pipeline Execution Finished Successfully!")


if __name__ == "__main__":
    main()
