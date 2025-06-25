# PEPP Simulation Engine

The **PEPP Simulation Engine** is a modular Python tool designed to simulate retirement wealth accumulation under the Pan-European Personal Pension Product (PEPP) framework. It uses **Geometric Brownian Motion (GBM)** to simulate asset returns and **Cholesky decomposition** to model correlations between assets.

---

## 📦 Project Structure

```plaintext
pepp-simulation-engine/
├── data/
│   ├── synthetic_inflation.csv         # Synthetic inflation index over time
│   └── synthetic_price_data.csv        # Historical synthetic asset prices
│
├── pepp_calculations/
│   ├── __init__.py
│   ├── multi_asset_simulation.py      # Main simulation engine class
│   ├── risk_metrics.py                # PEPP-style risk classification
│   └── utils.py                       # Helper: input conversion, reporting
│
├── simulation_methods/
│   ├── __init__.py
│   └── cholesky.py                    # Correlated return generation using Cholesky
│
├── main.py                            # Entry point to configure and run simulation
├── README.md                          # Project documentation (this file)
└── requirements.txt                   # Requirements file

```

---

## ✅ Features

- 📈 Simulates **multi-asset portfolios** using GBM dynamics
- 🔁 Introduces asset correlation via **Cholesky decomposition**
- 📊 Outputs inflation-adjusted wealth projections
- 🧮 Calculates PEPP risk/reward metrics
- 📁 Uses realistic synthetic price and inflation data

---

## 📥 Installation

Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

```


### Running in a virtual environment

With just two standard dependencies (pandas and numpy), running in a virtual environment is not a must. In case you want to do anyway, we like [Astral's UV](https://docs.astral.sh.uv), which can be utilized like this (after [installation]()):

```

```bash
uv venv --python=3.12
uv pip install -r requirements.txt
source .venv/bin/activate
```

---

## ▶️ How to Run

Run the simulation with:

```bash
python main.py
```

This will:

1. Load historical price and inflation data from the `data/` folder.
2. Estimate parameters required for Cholesky decomposition.
3. Simulate wealth evolution across thousands of Monte Carlo paths.
4. Display the results.

---

## 🔍 Key Modules

### `multi_asset_simulation.py`

Simulates correlated GBM return paths across multiple assets.

### `cholesky.py`

Computes and applies Cholesky decomposition to the covariance matrix to create correlated shocks.

### `risk_metrics.py`

Calculates the PEPP risk and reward metrics.

---

## 📊 Example Output

- Final portfolio values
- Summary statistics printed to console
- CSV export and visualizations can be added easily

---

## ✏️ Modifying the Main File

To customize the simulation (e.g., change the model, time horizon, weights, or contributions), edit the `main.py` script directly:

- Adjust the `returns`, `mu`, `sigma`, and `cov_matrix` logic to use alternative models like **GJR-GARCH**.
- Modify the contribution amount, number of simulations, or asset weights for different investment strategies.
- To implement GARCH-based models, replace the GBM logic inside `multi_asset_simulation.py` and update parameter preparation accordingly.

This structure is designed to make model substitution (e.g., replacing GBM with GJR-GARCH + copulas) straightforward and contained.
