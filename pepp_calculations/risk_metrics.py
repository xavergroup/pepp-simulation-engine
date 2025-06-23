"""
Risk and reward metrics for evaluating simulated capital accumulation outcomes.

Includes:
- Risk of not recouping contributions
- Expected shortfall conditional on loss
- Expected reward ratio based on the median path
"""

import numpy as np

def calculate_risk_of_not_recouping(accumulated_capital, total_inflation_adjusted_contributions):
        """
        Calculates the percentage of simulation paths where the final accumulated capital 
        fails to recoup the inflation-adjusted contributions.

        Args:
            accumulated_capital (np.ndarray): 1D array of final capital values from simulations.
            total_inflation_adjusted_contributions (float): Total contributions adjusted for inflation.

        Returns:
            float: Risk percentage (0–100) representing the share of simulations where 
                accumulated capital is less than contributions.
        """
        failures = np.sum(accumulated_capital < total_inflation_adjusted_contributions)
        return (failures / accumulated_capital.shape[0]) * 100

def calculate_expected_shortfall(accumulated_capital, total_inflation_adjusted_contributions):
    """
    Calculates the expected shortfall when the simulated capital fails to 
    recoup the inflation-adjusted contributions.

    The shortfall is defined as the mean loss (in percentage terms) across all
    simulations where the final capital is below the total contributions.

    Args:
        accumulated_capital (np.ndarray): 1D array of final capital values from simulations.
        total_inflation_adjusted_contributions (float): Total contributions adjusted for inflation.

    Returns:
        float: Expected shortfall (0–100), as a percentage of contributions. Returns 0 if no shortfalls occurred.
    """
    shortfalls = accumulated_capital - total_inflation_adjusted_contributions
    negative_shortfalls = shortfalls[shortfalls < 0]

    if len(negative_shortfalls) > 0:
        return (negative_shortfalls.mean() / total_inflation_adjusted_contributions) * 100

    return 0

def calculate_expected_rewards(accumulated_capital, total_inflation_adjusted_contributions):
    """
    Calculates the expected reward ratio based on the median simulated outcome.

    This metric reflects how much, on average, the median investor earns relative 
    to their total inflation-adjusted contributions.

    Args:
        accumulated_capital (np.ndarray): 1D array of final capital values from simulations.
        total_inflation_adjusted_contributions (float): Total contributions adjusted for inflation.

    Returns:
        float: Ratio of median accumulated capital to total contributions.
               A value >1 indicates capital growth; <1 indicates loss.
    """
    median_accumulated_capital = np.median(accumulated_capital)
    return median_accumulated_capital / total_inflation_adjusted_contributions