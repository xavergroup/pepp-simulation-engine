import numpy as np
from .risk_metrics import calculate_risk_of_not_recouping, calculate_expected_shortfall, calculate_expected_rewards

class MultiAssetSimulation:
    """
    Handles simulation and analysis of multi-asset portfolios over time.

    Integrates forward return generation, capital accumulation, inflation adjustment,
    contribution modeling, and final wealth/risk evaluation across defined simulation periods.
    """

    def __init__(self, data, returns, forwards_returns, n_assets, n_equity, contributions, periods, months, 
                 simulations, inflation_df, s0, weights=None, retirement_target_weights=None,
                 mu=None, sigma=None, cov_matrix=None, L=None, fee=100):
        """
        Initializes the MultiAssetSimulation class with portfolio parameters and simulation settings.

        Args:
            data (pd.DataFrame): Historical price data for assets.
            returns (pd.DataFrame): Historical asset returns.
            forwards_returns (object): Simulation method object with a simulate_monthly_returns() method (e.g., Cholesky-based).
            n_assets (int): Number of assets in the portfolio.
            n_equity (int): Number of equity assets in the portfolio.
            contributions (float): Monthly contributions to the portfolio.
            months (int): Total number of months in the simulation.
            periods (list[int]): Milestone periods (in years) for evaluating portfolio performance. Must be a subset of [10, 20, 30, 40].
            simulations (int): Number of Monte Carlo simulations.
            inflation_df (pd.DataFrame): Inflation rate data indexed by year.
            s0 (float): Initial portfolio value.
            weights (np.ndarray, optional): Initial asset weights. Defaults to None.
            retirement_target_weights (np.ndarray, optional): Target allocation for retirement phase. Defaults to None.
            mu (np.ndarray, optional): Expected returns for assets. Defaults to None.
            sigma (np.ndarray, optional): Volatility of assets. Defaults to None.
            cov_matrix (np.ndarray, optional): Covariance matrix of asset returns. Defaults to None.
            L (np.ndarray, optional): Cholesky decomposition of the covariance matrix. Defaults to None.
            fee (float, optional): Yearly transaction fee in basis points (bps). Defaults to 100 = 1%.
        """
        self.data = data
        self.returns = returns
        self.forwards_returns = forwards_returns
        self.n_assets = n_assets
        self.n_equity = n_equity
        self.contributions = contributions
        self.months = months
        self.periods = periods
        self.simulations = simulations
        self.inflation_df = inflation_df
        self.s0 = s0
       
        self.weights = weights if weights is not None else None
        self.retirement_target_weights = retirement_target_weights if retirement_target_weights is not None else None

        self._mu = mu
        self._sigma = sigma
        self._drift = self._mu - self._sigma**2 / 2 if (self._mu is not None and self._sigma is not None) else None
        self.cov_matrix = cov_matrix
        self.L = L
        
        self.fee = fee
    
    @property
    def weights(self):
        """Ensure weights are set before accessing them."""
        if self._weights is None:
            raise ValueError("Error: Portfolio weights are not set. Please assign weights before running the simulation.")
        
        return self._weights
    
    @weights.setter
    def weights(self, new_weights):
        """Assign weights only if valid."""
        if not isinstance(new_weights, (list, np.ndarray)):
            raise TypeError("Error: Weights must be a list or numpy array.")

        self._weights = np.array(new_weights, dtype=float)

    @property
    def retirement_target_weights(self):
        """Ensure retirement weights are set before accessing them."""
        if self._retirement_target_weights is None:
            raise ValueError("Error: Retirement target weights are not set. Please assign weights before running the simulation.")

        return self._retirement_target_weights

    @retirement_target_weights.setter
    def retirement_target_weights(self, new_weights):
        """Assign retirement weights only if valid."""
        if not isinstance(new_weights, (list, np.ndarray)):
            raise TypeError("Error: Retirement weights must be a list or numpy array.")
    
        self._retirement_target_weights = np.array(new_weights, dtype=float)

    @property
    def drift(self):
        """Ensure drift is only computed when mu and sigma exist."""
        if self._drift is None:
            raise ValueError("Error: Drift cannot be computed because mu and/or sigma is not set.")

        return self._drift

    @property
    def L(self):
        """Ensure Cholesky decomposition matrix is set before using it."""
        if self._L is None:
            raise ValueError("Error: Cholesky decomposition matrix (L) is not set. Ensure covariance matrix is computed.")

        return self._L

    @L.setter
    def L(self, new_L):
        """Assign Cholesky matrix if valid."""
        if not isinstance(new_L, np.ndarray):
            raise TypeError("Error: L must be a numpy array.")

        self._L = new_L
    
    @property
    def periods(self):
        """Ensure milestone periods are set before use."""
        if self._periods is None:
            raise ValueError("Error: Periods must be set before accessing them.")
        return self._periods

    @periods.setter
    def periods(self, value):
        """Validate that periods is a list of allowed values and does not exceed the simulation length."""
        allowed_periods = {10, 20, 30, 40}

        if not isinstance(value, list):
            raise TypeError("Error: periods must be a list of integers.")

        if not all(isinstance(p, int) for p in value):
            raise ValueError("Error: All entries in periods must be integers.")

        if not set(value).issubset(allowed_periods):
            raise ValueError(f"Error: periods must only contain a subset of {allowed_periods}.")

        if max(value) * 12 > self.months:
            raise ValueError(f"Error: Maximum period ({max(value)} years) exceeds total simulation length ({self.months} months).")

        self._periods = value

    def categorize_risk_of_not_recouping(self, risk_of_not_recouping):
        """
        Categorizes the risk of not recouping contributions using PEPP regulatory thresholds.

        Uses simulation periods to select the relevant threshold mapping.

        Args:
            risk_of_not_recouping (float): Risk percentage that contributions are not recouped.

        Returns:
            int: Risk category (1 = lowest risk, 4 = highest risk).
        """
        risk_categories = {
            40: [13.75, 16.55, 19.35],
            30: [17, 19.75, 22.55],
            20: [27, 29.25, 31.55],
            10: [36, 43.25, 50.55]
        }

        category = 4
        for period in self.periods:
            thresholds = risk_categories[period]
            if risk_of_not_recouping <= thresholds[0]:
                category = min(category, 1)
            elif risk_of_not_recouping <= thresholds[1]:
                category = min(category, 2)
            elif risk_of_not_recouping <= thresholds[2]:
                category = min(category, 3)

        return category

    def categorize_expected_shortfall(self, expected_shortfall):
        """
        Categorizes expected shortfall using PEPP thresholds per simulation period.

        Higher (less negative) values imply smaller shortfall and lower risk category.

        Args:
            expected_shortfall (float): Shortfall as a negative percentage of contributions.

        Returns:
            int: Shortfall category (1 = lowest risk, 4 = highest risk).
        """
        shortfall_categories = {
            40: [-20, -23, -26.5],
            30: [-17, -20.25, -23.55],
            20: [-13, -16.5, -20.1],
            10: [-8, -11.25, -14.55]
        }

        category = 4
        for period in self.periods:
            thresholds = shortfall_categories[period]
            if expected_shortfall > thresholds[0]:
                category = min(category, 1)
            elif expected_shortfall > thresholds[1]:
                category = min(category, 2)
            elif expected_shortfall > thresholds[2]:
                category = min(category, 3)

        return category

    def categorize_expected_rewards(self, rewards):
        """
        Categorizes expected rewards using PEPP thresholds for each simulation period.

        Higher expected reward implies better performance and a lower risk category.

        Args:
            rewards (float): Expected reward as ratio of capital to contributions.

        Returns:
            int: Rewards category (1 = best, 4 = lowest).
        """
        reward_categories = {
            40: [1.7, 2.03, 2.36],
            30: [1.3, 1.45, 1.61],
            20: [1.08, 1.165, 1.255],
            10: [0.93, 0.985, 1.045]
        }

        category = 4
        for period in self.periods:
            thresholds = reward_categories[period]
            if rewards <= thresholds[0]:
                category = min(category, 1)
            elif rewards <= thresholds[1]:
                category = min(category, 2)
            elif rewards <= thresholds[2]:
                category = min(category, 3)

        return category    
    
    def period_risk_indicator(self, risk_categories, shortfall_categories):
        """
        Combines risk and shortfall categories into a risk indicator value.

        The highest of the two values (risk or shortfall) is selected as the final risk indicator for the last period.

        Args:
            risk_categories (dict): Risk categories per period.
            shortfall_categories (dict): Shortfall categories per period.

        Returns:
            dict: Dictionary with one entry: {last_period: summary_risk_indicator}.
        """      
        summary_risk_indicators = {}

        last_period = self.periods[-1]

        combined_risk_category = max(risk_categories[last_period], shortfall_categories[last_period])
        summary_risk_indicators[last_period] = combined_risk_category

        return summary_risk_indicators
    
    def calculate_expected_wealth_scenarios(self, accumulated_capital):
        """"
        Computes PEPP-style wealth scenarios for each milestone period.

        Scenarios include:
        - Stress (5th percentile)
        - Unfavourable (15th percentile)
        - Median (50th percentile)
        - Favourable (85th percentile)

        Args:
            accumulated_capital (np.ndarray): Simulated capital matrix (shape: months x simulations).

        Returns:
            tuple: (stress, unfavourable, median, favourable) capital values at the last period.
        """
        indices = [(period * 12) - 1 for period in self.periods]

        favorable_scenario = {}
        best_estimate_scenario = {}
        unfavourable_scenario = {}
        stress_scenario = {}

        for i, period in enumerate(self.periods):

            capital_at_period_end = accumulated_capital[indices[i], :]
            fav = round(np.percentile(capital_at_period_end, 85))
            med = round(np.median(capital_at_period_end))
            unfav = round(np.percentile(capital_at_period_end, 15))
            stress = round(np.percentile(capital_at_period_end, 5))

            favorable_scenario[period] = fav
            best_estimate_scenario[period] = med
            unfavourable_scenario[period] = unfav
            stress_scenario[period] = stress

        return stress, unfav, med, fav
    
    def calculate_transaction_cost(self, accumulated_capital):
        """
        Calculates monthly transaction cost applied to accumulated capital.

        Assumes yearly fee in basis points is spread evenly across 12 months.

        Args:
            accumulated_capital (float): Capital value at which cost is calculated.

        Returns:
            float: Monthly transaction cost.
        """
        transaction_cost_bps_monthly = self.fee / 12
        transaction_cost_percentage = transaction_cost_bps_monthly / 10000
        transaction_cost = transaction_cost_percentage * accumulated_capital

        return transaction_cost
    
    def simulate_accumulated_capital(self):
        """"
        Simulates portfolio value evolution over time via Monte Carlo, including rebalancing,
        derisking, contributions, inflation, and transaction costs.

        Returns:
            tuple:
                - accumulated_capital_every_period (np.ndarray): Capital at each milestone period (shape: [len(periods), simulations]).
                - inflation_adjusted_contributions_every_period (np.ndarray): Cumulative contributions (inflation-adjusted) per period.
                - accumulated_capital (np.ndarray): Capital path for each month and simulation (shape: [months, simulations]).
        """
        #time and inflation setup
        current_date = self.returns.index[-1]
        cumulative_inflation_factor = 1.0

        #contributions & milestone tracking containers
        inflation_adjusted_contributions_array = []
        inflation_adjusted_contributions_every_period = []
        accumulated_capital_every_period = []

        #cost tracking
        yearly_list_costs = []
        yearly_list_acc_capital = []
        yearly_accumulated_costs = 0

        #time horizon and return simulation
        periods_in_months = np.array([p * 12 for p in self.periods])
        monthly_returns_simulations = self.forwards_returns.simulate_monthly_returns()

        #capital initialization
        accumulated_capital = np.zeros((self.months, self.simulations), dtype=np.float64)
        accumulated_capital_assets = np.zeros((self.months, self.n_assets, self.simulations), dtype=np.float64)
        accumulated_capital[0, :] = self.s0
        accumulated_capital_assets[0, :, :] = self.s0 * self.weights[:, np.newaxis]

        #weight initialization
        weights = np.zeros((self.months, self.n_assets, self.simulations), dtype=np.float64)
        weights[0, :, :] = self.weights[:, np.newaxis]

        #derisking setup
        first_month_derisking = periods_in_months[-2] + 1 if len(periods_in_months) > 1 else 1
        derisking_initialized = False

        for month in range(1, self.months):
            #advance simulation time
            current_date = self.forwards_returns.add_one_month(current_date)

            #get return and weight inputs
            returns_for_the_month = monthly_returns_simulations[month - 1, :, :]
            weights_for_the_month = weights[month - 1, :, :]

            #apply returns to update capital by asset and in total
            accumulated_capital_assets[month, :, :] = accumulated_capital_assets[month - 1, :, :] * (1 + returns_for_the_month)
            accumulated_capital[month, :] = np.sum(accumulated_capital_assets[month, :, :], axis=0)

            #update weights based on new NAV
            new_NAV = accumulated_capital[month, :]
            weights[month, :, :] = accumulated_capital_assets[month, :, :] / new_NAV

            #rebalancing logic based on 10% drift from initial weights
            current_equity_sum = np.sum(weights_for_the_month[:self.n_equity, :], axis=0)
            initial_equity_sum = np.sum(weights[0, :self.n_equity, :], axis=0)
            equity_deviation = np.abs((current_equity_sum - initial_equity_sum) / initial_equity_sum)

            current_bond_sum = np.sum(weights_for_the_month[self.n_equity:, :], axis=0)
            initial_bond_sum = np.sum(weights[0, self.n_equity:, :], axis=0)
            bond_deviation = np.abs((current_bond_sum - initial_bond_sum) / initial_bond_sum)

            rebalance_indices = np.logical_or(equity_deviation > 0.1, bond_deviation > 0.1)

            #apply transaction costs
            transaction_cost = self.calculate_transaction_cost(new_NAV)
            yearly_accumulated_costs += transaction_cost
            new_NAV -= transaction_cost

            #apply inflation and adjust contributions
            year_of_current_month = current_date.year
            annual_inflation_rate = self.inflation_df.loc[year_of_current_month, 'Inflation']
            monthly_inflation_rate = (1 + annual_inflation_rate) ** (1 / 12) - 1
            cumulative_inflation_factor *= (1 + monthly_inflation_rate)

            inflation_adjusted_contributions = self.contributions / cumulative_inflation_factor
            inflation_adjusted_contributions_array.append(inflation_adjusted_contributions)

            new_NAV += inflation_adjusted_contributions
            accumulated_capital[month, :] = new_NAV

            #apply rebalancing or derisking logic
            if month < first_month_derisking:
                weights[month, :, rebalance_indices] = weights[0, :, rebalance_indices]
            else:
                time_since_derisking_start = month - first_month_derisking
                total_derisking_period = self.months - first_month_derisking
                derisking_factor = time_since_derisking_start / total_derisking_period

                if not derisking_initialized:
                    initial_derisking_weights = weights[month - 1, :, :]
                    derisking_initialized = True

                weights[month, :, :] = (
                    (1 - derisking_factor) * initial_derisking_weights +
                    derisking_factor * self.retirement_target_weights[:, np.newaxis]
                )
                weights[month, :, :] /= np.sum(weights[month, :, :], axis=0)  # Normalize

            #reallocate updated NAV back to asset level
            for asset in range(self.n_assets):
                accumulated_capital_assets[month, asset, :] = new_NAV * weights[month, asset, :]

            #capture milestone results
            if month in periods_in_months - 1:
                accumulated_capital_every_period.append(accumulated_capital[month, :])
                inflation_adjusted_contributions_every_period.append(np.sum(inflation_adjusted_contributions_array))

            #log annual metrics
            if month % 12 == 0 or month == periods_in_months[-1] - 1:
                yearly_list_costs.append(yearly_accumulated_costs)
                yearly_list_acc_capital.append(accumulated_capital[month, :])
                yearly_accumulated_costs = 0

        return (
            np.array(accumulated_capital_every_period),
            np.array(inflation_adjusted_contributions_every_period),
            accumulated_capital
        )

    def run_simulation(self):
        """
        Executes the full Monte Carlo simulation for the portfolio, 
        applying risk and reward evaluation based on capital accumulation.

        Returns:
            dict: Simulation output containing:
                - accumulated_capital (np.ndarray): Full capital time series.
                - projected_sum_of_contributions (float): Total contributions adjusted for inflation.
                - stress (float): 5th percentile wealth outcome (stress scenario).
                - pess (float): 15th percentile wealth outcome (unfavourable scenario).
                - med (float): 50th percentile wealth outcome (median).
                - opt (float): 85th percentile wealth outcome (favourable scenario).
                - risk_of_not_recouping (float): Probability (%) of not recovering contributions.
                - shortfall (float): Expected loss if contributions are not recovered.
                - rewards (float): Median reward ratio (capital / contributions).
                - period_risk_ind (dict): PEPP risk indicator per period.
        """
        
        #run simulation core
        accumulated_capital_every_period, inflation_adjusted_contributions_every_period, accumulated_capital = self.simulate_accumulated_capital()
        last_period = self.periods[-1]

        #get capital and contributions for final evaluation period
        inflation_contrib_map = dict(zip(self.periods, inflation_adjusted_contributions_every_period))
        capital_map = dict(zip(self.periods, accumulated_capital_every_period))
        current_inflation_contributions = inflation_contrib_map[last_period]
        current_accumulated_capital = capital_map[last_period]

        #risk metrics
        risk_of_not_recouping = calculate_risk_of_not_recouping(current_accumulated_capital, current_inflation_contributions)
        risk_of_not_recouping_cat = self.categorize_risk_of_not_recouping(risk_of_not_recouping)

        shortfall = calculate_expected_shortfall(current_accumulated_capital, current_inflation_contributions)
        shortfall_cat = self.categorize_expected_shortfall(shortfall)

        rewards = calculate_expected_rewards(current_accumulated_capital, current_inflation_contributions)
        reward_cat = self.categorize_expected_rewards(rewards)

        #combine into PEPP-style risk indicator
        period_risk_ind = self.period_risk_indicator(
            {last_period: risk_of_not_recouping_cat},
            {last_period: shortfall_cat}
        )

        #wealth scenarios
        stress, pess, med, opt = self.calculate_expected_wealth_scenarios(accumulated_capital)

        return {
            "accumulated_capital": accumulated_capital,
            "projected_sum_of_contributions": current_inflation_contributions,
            "stress": stress,
            "pess": pess,
            "med": med,
            "opt": opt,
            "risk_of_not_recouping": risk_of_not_recouping,
            "shortfall": shortfall,
            "rewards": rewards,
            "period_risk_ind": period_risk_ind
        }
