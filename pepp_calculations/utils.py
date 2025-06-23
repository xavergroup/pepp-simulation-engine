import numpy as np
from numpy.linalg import cholesky

def prepare_parameters_cholesky(returns, data_frequency: str = "daily"):
    """
    Prepares mu, sigma, cov_matrix, L, and drift in monthly scale from log returns.

    Args:
        returns (pd.DataFrame): Log return DataFrame.
        data_frequency (str): 'daily', 'weekly', or 'monthly'.

    Returns:
        tuple: (mu_monthly, sigma_monthly, cov_monthly, L_monthly, drift_monthly)
    """
    freq_map = {
        "daily": 252,
        "weekly": 52,
        "monthly": 12
    }

    if data_frequency.lower() not in freq_map:
        raise ValueError(f"Unsupported data frequency: {data_frequency}. Choose from {list(freq_map.keys())}.")

    freq = freq_map[data_frequency.lower()]

    #estimate from raw returns
    mu_daily = returns.mean()
    sigma_daily = returns.std()
    cov_daily = returns.cov()

    #convert to monthly scale (log returns are additive)
    mu = mu_daily * freq / 12
    sigma = sigma_daily * np.sqrt(freq / 12)
    cov_matrix = cov_daily * freq / 12

    #dDrift for GBM
    drift = mu - 0.5 * sigma ** 2
    L = cholesky(cov_matrix)

    return mu, sigma, cov_matrix, L, drift


def print_summary(label, result):
    """
    Prints a summary of PEPP simulation results, including:
    - Risk of not recouping
    - Expected shortfall
    - Expected rewards
    - Wealth scenarios
    - Period risk indicator

    Args:
        label (str): Label for the simulation result (e.g., '10Y Results').
        result (dict): Dictionary containing keys:
            'risk_of_not_recouping', 'shortfall', 'rewards',
            'stress', 'unfavourable', 'median', 'favourable',
            and 'period_risk_indicator'.
    """
    print(f"\n=== {label} ===")

    # Risk metrics
    print("Risk Metrics:")
    print(f"  Risk of Not Recouping: {result['risk_of_not_recouping']:.2f}%")
    print(f"  Expected Shortfall: {result['shortfall']:.2f}%")
    print(f"  Expected Rewards: {result['rewards']:.2f}x")

    # Wealth scenarios
    print("\nWealth Scenarios:")
    print(f"  Stress: {result['stress']}")
    print(f"  Unfavourable: {result['pess']}")
    print(f"  Median: {result['med']}")
    print(f"  Favourable: {result['opt']}")

    # Risk indicator
    print("\nPeriod Risk Indicator:")
    print(f"  {result['period_risk_ind']}")
