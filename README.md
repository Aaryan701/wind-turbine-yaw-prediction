# Deep Learning Wind Turbine Yaw Prediction System with Floral Optimization

An end-to-end industrial-grade Machine Learning and Deep Learning pipeline designed to predict wind turbine yaw angles and rotational tracking dynamics using high-frequency SCADA telemetry.

---

## 🌟 Key Features

1. **SCADA Telemetry & Aerodynamic Dynamics Engine**:
   - Models realistic wind turbine aerodynamics: Ornstein-Uhlenbeck wind speed drifts, stochastic wind direction fluctuations, active power degradation ($\cos^2(\gamma)$ yaw loss), and nacelle deadband controller hysteresis ($\pm 7^\circ$).
   - Angular-safe data cleaning and vector-based Cartesian interpolation ($\sin / \cos$ decomposition to prevent $359^\circ \leftrightarrow 1^\circ$ interpolation singularities).

2. **Temporal Feature Engineering**:
   - Cyclical trigonometric transformations for circular continuity.
   - Rotational velocities ($\Delta \theta / \Delta t$) and angular accelerations.
   - Multi-scale rolling statistical aggregates (turbulence intensity indicators).
   - Non-leaking sequence sliding-window matrix generation.

3. **Hybrid Deep Learning Architecture (`TemporalConvGRU`)**:
   - **1D Temporal Convolution**: Extracts micro-turbulence signatures and wind gust shockwaves across historical windows.
   - **Bidirectional GRU / LSTM**: Captures macro wind drifts and rotational inertia.
   - **Temporal Attention Layer**: Focuses dynamically on critical turning points and deadband boundary crossings.
   - **Custom Angular Loss Functions (`CircularMSELoss`, `CircularMAELoss`)**: Geodesic circular distance penalties accounting for circular topology.

4. **Flower Pollination Algorithm (FPA) Metaheuristic Optimizer**:
   - Implements Xin-She Yang's Flower Pollination Algorithm.
   - Global pollination via cross-pollination driven by **Lévy flights** (Mantegna's algorithm) to avoid local minima trapping.
   - Local pollination via abiotic neighbor pollen transfer for fine-grained hyperparameter exploitation.
   - Tunes convolutional filters, GRU latent dimensions, network depth, dropout regularization, and learning rate.

---

## 📂 Project Architecture

```
wind_turbine_yaw_system/
├── requirements.txt         # Required Python packages
├── scada_simulator.py       # SCADA telemetry generator & physics simulator
├── eda_visualizer.py        # EDA, angular interpolation, Seaborn/Matplotlib suite
├── feature_engineering.py   # Cyclical encoding, dynamics, sliding windows
├── models.py                # TemporalConvGRU, Circular MSE/MAE Loss, metrics
├── floral_optimizer.py      # Flower Pollination Algorithm (FPA) tuner
├── main.py                  # End-to-end orchestrator & comparative benchmark
├── eda_plots/               # Exported high-resolution EDA figures
└── results_plots/           # Exported benchmark & convergence figures
```

---

## 🚀 Quickstart

### 1. Installation
Ensure Python 3.9+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Complete End-to-End Pipeline
```bash
python main.py
```

This single command will:
1. Ingest/generate SCADA telemetry.
2. Execute automated EDA and export publication-ready plots to `eda_plots/`.
3. Construct sliding window sequences and partition chronologically without data leakage.
4. Train the baseline Deep Learning model.
5. Run the Flower Pollination Algorithm to discover optimal hyperparameters.
6. Train the optimized model and evaluate on unseen test data.
7. Print the benchmark performance matrix and save tracking plots to `results_plots/`.

---

## 📐 Mathematical Formulations

### 1. The Circular Boundary Singularity
Standard MSE evaluates error as $(\hat{\theta} - \theta)^2$. If $\hat{\theta} = 1^\circ$ and $\theta = 359^\circ$:
$$\text{Standard MSE} = (1 - 359)^2 = 128,164$$
Whereas the geodesic angular distance along the circle is only $2^\circ$!

### 2. Geodesic Circular Distance Loss
$$\Delta \theta = ((\hat{\theta} - \theta + 180^\circ) \pmod{360^\circ}) - 180^\circ$$
$$\mathcal{L}_{\text{CircMSE}} = \frac{1}{N} \sum_{i=1}^N (\Delta \theta_i)^2$$
$$\mathcal{L}_{\text{CircMAE}} = \frac{1}{N} \sum_{i=1}^N |\Delta \theta_i|$$

### 3. Flower Pollination Algorithm (FPA)
- **Global Pollination (Biotic / Cross-Pollination)**:
  $$x_i^{t+1} = x_i^t + \gamma L (g^* - x_i^t)$$
  where $g^*$ is the current best pollen gamete, and $L \sim \text{Lévy}(\lambda)$.
- **Local Pollination (Abiotic / Neighbor Pollination)**:
  $$x_i^{t+1} = x_i^t + \epsilon (x_j^t - x_k^t)$$
  where $\epsilon \sim U(0, 1)$, and $x_j^t, x_k^t$ are random pollen gametes of the same species.
- **Switch Probability**:
  $$p = 0.8$$
