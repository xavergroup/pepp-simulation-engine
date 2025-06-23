from abc import ABC, abstractmethod
import numpy as np

class ReturnSimulation(ABC):
    """
    Abstract base class for simulating asset return time series.
    Subclasses must implement the `simulate_monthly_returns` method to 
    produce synthetic return data, typically for use in portfolio analysis 
    or Monte Carlo simulations.
    """

    @abstractmethod
    def simulate_monthly_returns(self) -> np.ndarray:
        """
        Generate simulated monthly returns.

        Returns:
            np.ndarray: A 3D array of shape (months, n_assets, simulations) 
                        containing simulated monthly returns.
        """
        pass
