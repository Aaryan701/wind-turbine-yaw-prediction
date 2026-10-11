"""
floral_optimizer.py
-------------------
Flower Pollination Algorithm (FPA) for Deep Learning Hyperparameter Optimization.
Based on Xin-She Yang (2012): "Flower Pollination Algorithm for Global Optimization".

Features:
- Global Pollination with Lévy Flight jumps (Mantegna's algorithm)
- Local Pollination via abiotic neighbor cross-pollination
- Boundary constraint clipping
- Fast proxy evaluation with early stopping for neural network fitness calculation
- Convergence tracking and benchmark comparison
"""

import math
import copy
from typing import Dict, Any, List, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import TemporalConvGRU, CircularMSELoss, CircularMAELoss, compute_angular_metrics


class FlowerPollinationOptimizer:
    """
    Flower Pollination Algorithm (FPA) tailored for Deep Learning hyperparameter tuning.
    """
    def __init__(
        self,
        n_flowers: int = 8,
        max_generations: int = 6,
        switch_prob: float = 0.8,
        levy_lambda: float = 1.5,
        random_state: int = 42
    ):
        """
        Parameters:
            n_flowers (int): Population size (number of pollen gametes / candidate models).
            max_generations (int): Number of optimization iterations / generations.
            switch_prob (float): Probability p for global vs local pollination (default 0.8).
            levy_lambda (float): Scaling exponent for Lévy distribution (typically 1.5).
            random_state (int): Random seed for reproducibility.
        """
        self.n_flowers = n_flowers
        self.max_generations = max_generations
        self.p = switch_prob
        self.lamb = levy_lambda
        self.rng = np.random.RandomState(random_state)
        
        # Hyperparameter search space boundaries: [lower, upper, type]
        # [0] conv_channels: int in [16, 64]
        # [1] gru_hidden_dim: int in [32, 128]
        # [2] gru_layers: int in [1, 3]
        # [3] dropout: float in [0.05, 0.45]
        # [4] log10_lr: float in [-4.0, -2.0]  --> lr = 10^val
        self.bounds = np.array([
            [16.0, 64.0],     # conv_channels
            [32.0, 128.0],    # gru_hidden_dim
            [1.0, 3.0],       # gru_layers
            [0.05, 0.45],     # dropout
            [-4.0, -2.0]      # log10(learning_rate)
        ])
        
        # Precompute Mantegna constant for Levy flight
        num = math.gamma(1 + self.lamb) * math.sin(math.pi * self.lamb / 2)
        den = math.gamma((1 + self.lamb) / 2) * self.lamb * (2 ** ((self.lamb - 1) / 2))
        self.sigma = (num / den) ** (1 / self.lamb)
        
        self.best_solution: np.ndarray = None
        self.best_fitness: float = float("inf")
        self.convergence_history: List[float] = []

    def _sample_levy_flight(self) -> np.ndarray:
        """
        Generates step size vector using Mantegna's algorithm for Levy stable distribution.
        """
        dim = len(self.bounds)
        u = self.rng.normal(0, self.sigma, size=dim)
        v = self.rng.normal(0, 1, size=dim)
        step = u / (np.abs(v) ** (1 / self.lamb))
        return step

    def _decode_hyperparameters(self, vector: np.ndarray) -> Dict[str, Any]:
        """
        Maps continuous optimization vector into discrete model hyperparameters.
        """
        return {
            "conv_channels": int(np.clip(np.round(vector[0]), 16, 64)),
            "gru_hidden_dim": int(np.clip(np.round(vector[1]), 32, 128)),
            "gru_layers": int(np.clip(np.round(vector[2]), 1, 3)),
            "dropout": float(np.clip(vector[3], 0.05, 0.45)),
            "learning_rate": float(10 ** np.clip(vector[4], -4.0, -2.0))
        }

    def _evaluate_candidate(
        self,
        params: Dict[str, Any],
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        proxy_epochs: int = 4,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu")
    ) -> float:
        """
        Trains candidate model for a few proxy epochs and evaluates Validation Circular MAE.
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        input_dim = X_train.shape[2]
        model = TemporalConvGRU(
            input_dim=input_dim,
            conv_channels=params["conv_channels"],
            gru_hidden_dim=params["gru_hidden_dim"],
            gru_layers=params["gru_layers"],
            dropout=params["dropout"]
        ).to(device)
        
        criterion = CircularMSELoss()
        mae_eval = CircularMAELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=1e-4)
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(proxy_epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                
        # Validation Evaluation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(device))
            val_circ_mae = mae_eval(val_preds, y_val.to(device)).item()
            
        return val_circ_mae

    def optimize(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        proxy_epochs: int = 4,
        batch_size: int = 32,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[Dict[str, Any], List[float]]:
        """
        Executes Flower Pollination metaheuristic optimization loop.
        """
        dim = len(self.bounds)
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        
        print("=" * 70)
        print("FLOWER POLLINATION ALGORITHM (FPA) HYPERPARAMETER TUNING")
        print("=" * 70)
        print(f"Population (Flowers) : {self.n_flowers}")
        print(f"Max Generations      : {self.max_generations}")
        print(f"Switch Probability p : {self.p} (Global: {self.p*100:.0f}%, Local: {(1-self.p)*100:.0f}%)")
        print("-" * 70)

        # 1. Initialize Flower Population uniformly across boundary ranges
        population = self.rng.uniform(low, high, size=(self.n_flowers, dim))
        fitness = np.zeros(self.n_flowers)

        print("Initializing & evaluating base floral population...")
        for i in range(self.n_flowers):
            hparams = self._decode_hyperparameters(population[i])
            fitness[i] = self._evaluate_candidate(
                hparams, train_data, val_data, proxy_epochs=proxy_epochs, batch_size=batch_size, device=device
            )
            print(f"  Flower #{i+1:02d} | Params: {hparams} | Val Circ MAE: {fitness[i]:.3f}°")

        # Find best initial flower g*
        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = population[best_idx].copy()
        self.convergence_history.append(self.best_fitness)

        print(f"Initial Best Floral Fitness: {self.best_fitness:.3f}°\n")

        # 2. Iterative Pollination Generations
        gamma = 0.1  # Step scaling factor
        for gen in range(1, self.max_generations + 1):
            for i in range(self.n_flowers):
                rand_val = self.rng.rand()
                
                if rand_val < self.p:
                    # RULE 1: Global Pollination (Biotic / Cross-pollination via Levy Flight)
                    # Pollen carried by pollinators over long distances:
                    # X_i^{t+1} = X_i^t + gamma * L * (g* - X_i^t)
                    levy_step = self._sample_levy_flight()
                    candidate = population[i] + gamma * levy_step * (self.best_solution - population[i])
                else:
                    # RULE 2: Local Pollination (Abiotic / Neighbor pollination)
                    # Pollen transfer between two random neighbor flowers j and k of the same species:
                    # X_i^{t+1} = X_i^t + epsilon * (X_j^t - X_k^t)
                    j, k = self.rng.choice(self.n_flowers, size=2, replace=False)
                    epsilon = self.rng.rand(dim)
                    candidate = population[i] + epsilon * (population[j] - population[k])

                # Boundary constraint enforcement (clip to domain envelope)
                candidate = np.clip(candidate, low, high)
                cand_hparams = self._decode_hyperparameters(candidate)
                
                # Evaluate fitness
                cand_fitness = self._evaluate_candidate(
                    cand_hparams, train_data, val_data, proxy_epochs=proxy_epochs, batch_size=batch_size, device=device
                )

                # Greedy selection: if new flower yields better nectar / lower loss, replace
                if cand_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = cand_fitness

                    # Update global best if superior
                    if cand_fitness < self.best_fitness:
                        self.best_fitness = cand_fitness
                        self.best_solution = candidate.copy()

            self.convergence_history.append(self.best_fitness)
            print(f"Gen [{gen:02d}/{self.max_generations:02d}] Completed | Current Best Val Circ MAE: {self.best_fitness:.3f}°")

        best_params = self._decode_hyperparameters(self.best_solution)
        print("=" * 70)
        print("FLORAL OPTIMIZATION CONVERGED!")
        print(f"Optimal Hyperparameters : {best_params}")
        print(f"Best Validation Circ MAE : {self.best_fitness:.3f}°")
        print("=" * 70)
        
        return best_params, self.convergence_history


if __name__ == "__main__":
    dummy_x_train = torch.randn(100, 15, 8)
    dummy_y_train = torch.randn(100) * 180.0 % 360.0
    dummy_x_val = torch.randn(30, 15, 8)
    dummy_y_val = torch.randn(30) * 180.0 % 360.0
    
    optimizer = FlowerPollinationOptimizer(n_flowers=3, max_generations=2)
    best_p, history = optimizer.optimize((dummy_x_train, dummy_y_train), (dummy_x_val, dummy_y_val), proxy_epochs=1)
    print("Optimization test complete. Best parameters:", best_p)
