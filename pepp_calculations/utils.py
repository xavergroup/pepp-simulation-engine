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
