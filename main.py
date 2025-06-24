import numpy as np
import pandas as pd
from numpy.linalg import cholesky
from pepp_calculations import MultiAssetSimulation, print_summary, prepare_parameters_cholesky
from simulation_methods import CholeskyMethod

if __name__ == "__main__":
        
    #load data
    data_path = 'data/synthetic_price_data.csv'
    data = pd.read_csv(data_path, parse_dates=True, index_col='Date')
    data_frequency = 'daily'  #dataset has daily frequency

    #basic setup
    n_assets = len(data.columns)
    n_equity = 4
    returns = np.log(data / data.shift(1)).dropna()
    mu, sigma, cov_matrix, L, drift = prepare_parameters_cholesky(returns, "daily")

    simulations = 25000
    contributions = 100
    s0 = 100
    fee = 100
    initial_weights = [0.8, 0.2]
    final_weights = [0.6, 0.4]

    #inflation setup
    inflation_path = 'data/synthetic_inflation.csv'
    inflation_df = pd.read_csv(inflation_path, index_col=0)
    inflation_df.index = inflation_df.index.astype(int)

    # ---------- Simulation for 10 Years ----------
    periods_10 = [10]
    months_10 = 12 * max(periods_10)
    cholesky_simulator_10y = CholeskyMethod(simulations, months_10, n_assets, drift, L)
    simulator_10y = MultiAssetSimulation(
        data=data,
        returns=returns,
        forwards_returns=cholesky_simulator_10y,
        n_assets=n_assets,
        n_equity=n_equity,
        contributions=contributions,
        periods=periods_10,
        months=months_10,
        simulations=simulations,
        inflation_df=inflation_df,
        s0=s0,
        weights=initial_weights,
        retirement_target_weights=final_weights,
        mu=mu,
        sigma=sigma,
        cov_matrix=cov_matrix,
        L=L,
        fee=fee
    )
    result_10y = simulator_10y.run_simulation()

    # ---------- Simulation for 20 Years ----------
    periods_20 = [10, 20]
    months_20 = 12 * max(periods_20)
    cholesky_simulator_20y = CholeskyMethod(simulations, months_20, n_assets, drift, L)
    simulator_20y = MultiAssetSimulation(
        data=data,
        returns=returns,
        forwards_returns=cholesky_simulator_20y,
        n_assets=n_assets,
        n_equity=n_equity,
        contributions=contributions,
        periods=periods_20,
        months=months_20,
        simulations=simulations,
        inflation_df=inflation_df,
        s0=s0,
        weights=initial_weights,
        retirement_target_weights=final_weights,
        mu=mu,
        sigma=sigma,
        cov_matrix=cov_matrix,
        L=L,
        fee=fee
    )
    result_20y = simulator_20y.run_simulation()

    # ---------- Simulation for 30 Years ----------
    periods_30 = [10, 20, 30]
    months_30 = 12 * max(periods_30)
    cholesky_simulator_30y = CholeskyMethod(simulations, months_30, n_assets, drift, L)
    simulator_30y = MultiAssetSimulation(
        data=data,
        returns=returns,
        forwards_returns=cholesky_simulator_30y,
        n_assets=n_assets,
        n_equity=n_equity,
        contributions=contributions,
        periods=periods_30,
        months=months_30,
        simulations=simulations,
        inflation_df=inflation_df,
        s0=s0,
        weights=initial_weights,
        retirement_target_weights=final_weights,
        mu=mu,
        sigma=sigma,
        cov_matrix=cov_matrix,
        L=L,
        fee=fee
    )
    result_30y = simulator_30y.run_simulation()

    # ---------- Simulation for 40 Years ----------
    periods_40 = [10, 20, 30, 40]
    months_40 = 12 * max(periods_40)
    cholesky_simulator_40y = CholeskyMethod(simulations, months_40, n_assets, drift, L)
    simulator_40y = MultiAssetSimulation(
        data=data,
        returns=returns,
        forwards_returns=cholesky_simulator_40y,
        n_assets=n_assets,
        n_equity=n_equity,
        contributions=contributions,
        periods=periods_40,
        months=months_40,
        simulations=simulations,
        inflation_df=inflation_df,
        s0=s0,
        weights=initial_weights,
        retirement_target_weights=final_weights,
        mu=mu,
        sigma=sigma,
        cov_matrix=cov_matrix,
        L=L,
        fee=fee
    )
    result_40y = simulator_40y.run_simulation()

    #print summary of results
    print("\nFinished all simulations.")
    print_summary("10Y Results", result_10y)
    print_summary("20Y Results", result_20y)
    print_summary("30Y Results", result_30y)
    print_summary("40Y Results", result_40y)
