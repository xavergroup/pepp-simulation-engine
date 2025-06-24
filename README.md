# pepp-simulation-engine
Pepp Simulation Engine
# PEPP Simulation Engine

The **PEPP Simulation Engine** is a Python-based tool designed to simulate retirement investment scenarios under the Pan-European Personal Pension Product (PEPP) framework. It leverages Geometric Brownian Motion (GBM) for return simulation and uses Cholesky decomposition to model correlations between multiple assets.

---

## 🚀 Features

- 📈 Simulate asset returns using **Geometric Brownian Motion (GBM)**
- 🔁 Incorporate realistic asset correlations via **Cholesky decomposition**
- 📊 Track performance metrics including final portfolio value and inflation-adjusted outcomes
- ⚙️ Flexible configuration: change contributions, asset weights, and simulation horizon
- 📦 Modular codebase for extension and experimentation

---

## 📁 Project Structure

```plaintext
.
├── main.py                    # Entry point for running the simulation
├── base_simulator.py          # Abstract base class for simulation strategies
├── multi_asset_simulation.py  # Implements GBM-based multi-asset simulation
├── cholesky.py                # Handles Cholesky-based correlation of shocks
├── risk_metrics.py            # Provides metrics like max drawdown, final wealth
├── utils.py                   # Helper functions (e.g., logging, inflation adjustment)
├── __init__.py                # Module initialization
└── README.md                  # This file
