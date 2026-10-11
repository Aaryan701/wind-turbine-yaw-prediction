"""
models.py
---------
Deep Learning Architectures and Custom Angular Loss Functions for Wind Turbine Yaw Prediction.
Includes:
- Custom Circular MSE & Circular MAE Loss functions (handling 0° <-> 360° topology)
- TemporalConvGRU: Hybrid 1D-CNN + Bidirectional GRU + Self-Attention mechanism
- Baseline Vanilla LSTM for benchmark comparison
- Comprehensive angular evaluation metrics (Circular MAE, Circular RMSE, R²)
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score


# -------------------------------------------------------------------------
# Custom Angular Loss Functions
# -------------------------------------------------------------------------

class CircularMSELoss(nn.Module):
    """
    Circular Mean Squared Error Loss.
    Standard MSE fails dramatically on angular data because the distance between
    1 deg and 359 deg is penalized as (358)^2 = 128,164, whereas the true physical
    angular distance on a circle is only 2 deg.
    
    This loss maps angular errors into the geodesic range [-180, 180] degrees:
        error = ((y_pred - y_true + 180) % 360) - 180
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        # Geodesic angular difference in degrees
        angular_diff = torch.remainder(y_pred - y_true + 180.0, 360.0) - 180.0
        return torch.mean(angular_diff ** 2)


class CircularMAELoss(nn.Module):
    """
    Circular Mean Absolute Error Loss in degrees.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        angular_diff = torch.remainder(y_pred - y_true + 180.0, 360.0) - 180.0
        return torch.mean(torch.abs(angular_diff))


# -------------------------------------------------------------------------
# Temporal Attention Layer
# -------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Attention pooling over the temporal sequence dimension.
    Learns to dynamically weight historical time steps based on wind gust significance.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs: torch.Tensor) -> torch.Tensor:
        # rnn_outputs: [batch_size, seq_len, hidden_dim]
        scores = self.attn_weights(rnn_outputs)  # [batch_size, seq_len, 1]
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # [batch_size, hidden_dim]
        return context


# -------------------------------------------------------------------------
# Deep Learning Model Architectures
# -------------------------------------------------------------------------

class TemporalConvGRU(nn.Module):
    """
    Hybrid 1D CNN + Bidirectional GRU + Temporal Attention Network.
    - Conv1D filters capture high-frequency gust fluctuations and turbulent shifts.
    - Bidirectional GRU models forward/backward dependencies in yaw motion.
    - Attention focuses on pivotal deadband transitions.
    """
    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 32,
        gru_hidden_dim: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        # Conv1D expects shape [batch, channels, seq_len]
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        
        self.attention = TemporalAttention(hidden_dim=gru_hidden_dim * 2)
        
        self.fc_head = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, gru_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        x_trans = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        conv_out = self.conv(x_trans)  # [batch_size, conv_channels, seq_len]
        conv_out = conv_out.transpose(1, 2)  # [batch_size, seq_len, conv_channels]
        
        gru_out, _ = self.gru(conv_out)  # [batch_size, seq_len, gru_hidden_dim * 2]
        context = self.attention(gru_out)  # [batch_size, gru_hidden_dim * 2]
        
        out = self.fc_head(context)  # [batch_size, 1]
        # Keep raw output in degrees; circular loss handles the [0, 360) topology
        return out.squeeze(-1)


class VanillaLSTM(nn.Module):
    """
    Standard Baseline LSTM for comparison.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


# -------------------------------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------------------------------

def compute_angular_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes rigorous circular and standard metrics for yaw angle evaluation.
    
    Parameters:
        y_true (np.ndarray): True yaw angles in degrees [0, 360).
        y_pred (np.ndarray): Predicted yaw angles in degrees.
        
    Returns:
        Dict[str, float]: Circular MAE, Circular RMSE, Standard MAE, Standard RMSE, R^2.
    """
    y_true_mod = y_true % 360.0
    y_pred_mod = y_pred % 360.0
    
    # Geodesic angular difference in [-180, 180] deg
    angular_diff = (y_pred_mod - y_true_mod + 180.0) % 360.0 - 180.0
    
    circ_mae = float(np.mean(np.abs(angular_diff)))
    circ_rmse = float(np.sqrt(np.mean(angular_diff ** 2)))
    
    # Standard metrics without circular correction (demonstrating why circular metric is crucial)
    std_mae = float(np.mean(np.abs(y_pred_mod - y_true_mod)))
    std_rmse = float(np.sqrt(np.mean((y_pred_mod - y_true_mod) ** 2)))
    
    # Cosine alignment correlation score (1.0 = perfect alignment)
    cos_similarity = float(np.mean(np.cos(np.radians(angular_diff))))
    
    # Continuous R^2
    r2 = float(r2_score(y_true_mod, y_pred_mod))
    
    return {
        "Circular_MAE_deg": round(circ_mae, 3),
        "Circular_RMSE_deg": round(circ_rmse, 3),
        "Standard_MAE_deg": round(std_mae, 3),
        "Standard_RMSE_deg": round(std_rmse, 3),
        "Cosine_Alignment": round(cos_similarity, 4),
        "R2_Score": round(r2, 4)
    }


if __name__ == "__main__":
    # Smoke test loss function with 1 deg vs 359 deg
    loss_fn = CircularMSELoss()
    pred = torch.tensor([1.0, 10.0])
    true = torch.tensor([359.0, 10.0])
    loss = loss_fn(pred, true)
    print("Circular MSE test for (1 deg, 359 deg):", loss.item(), "(Expected: 2.0 degrees squared / 2 = 2.0)")
    
    # Smoke test model
    dummy_input = torch.randn(8, 20, 12)  # batch 8, seq 20, feat 12
    model = TemporalConvGRU(input_dim=12)
    output = model(dummy_input)
    print("TemporalConvGRU output shape:", output.shape)
