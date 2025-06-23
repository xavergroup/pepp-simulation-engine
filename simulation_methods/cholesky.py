import numpy as np
import pandas as pd
from .base_simulator import ReturnSimulation

class CholeskyMethod(ReturnSimulation):
    """
    Class for simulating correlated asset returns using the Cholesky decomposition method.
    """

    def __init__(self, simulations, months, n_assets, drift, L):
        """
        Initializes the Cholesky simulation parameters.

        Args:
            simulations (int): Number of Monte Carlo simulations to run.
            months (int): Number of time periods (e.g., months) in each simulation.
            n_assets (int): Number of assets for which returns are simulated.
            drift (np.ndarray): A 1D Series of length n_assets containing the constant drift term per asset.
            L (np.ndarray): The Cholesky decomposition of the covariance matrix, shape (n_assets, n_assets), used to 
                            create correlated returns.
        """
        self.simulations = simulations
        self.months = months
        self.n_assets = n_assets
        self.drift = drift
        self.L = L

    @staticmethod
    def add_one_month(date):
        """
        Add one month to the given date.

        Args:
            date (pd.Timestamp): Input date.

        Returns:
            pd.Timestamp: Date incremented by one month.
        """
        return date + pd.offsets.MonthEnd(1)

    def generate_random_z(self):
        """
        Generates a matrix of random values drawn from a standard normal distribution.

        Returns:
            np.ndarray: A 2D array of shape (months, n_assets) with i.i.d. standard normal samples.

        """
        return np.random.normal(size=(self.months, self.n_assets))

    def simulate_monthly_returns(self):
        """
        Simulates monthly returns using the Cholesky decomposition of a covariance matrix
        to generate correlated asset returns from standard normal innovations.

        Returns:
            np.ndarray: A 3D array of shape (months, n_assets, simulations) containing
                        the simulated monthly returns for each asset and simulation path.

        """
        monthly_returns_simulations = np.zeros((self.months, self.n_assets, self.simulations))

        for i in range(self.simulations):

            np.random.seed(i)
            z = self.generate_random_z()            
            correlated_random_walk = np.dot(z, self.L.T)
            drift_extended = np.tile(self.drift.values, (self.months, 1))
            monthly_returns = correlated_random_walk + drift_extended
            monthly_returns_simulations[:, :, i] = monthly_returns

        return monthly_returns_simulations

