# factor_correlation_analyzer.py

import pandas as pd


class FactorCorrelationAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        factors: list[str],
        method: "pearson" | "spearman" = "pearson",
        threshold=0.02,
        rolling_window=None,
    ):
        """
        Single target with multiple factors analyzer

        :param data: pandas DataFrame containing the dataset.
        :param target_column: String name of the target variable column.
        :param factors: List of string names of factor columns.
        :param method: Correlation method - 'pearson' or 'spearman'.
        :param threshold: Minimum correlation threshold to consider a factor significant.
        :param rolling_window: Window size for rolling correlation; if None, computes static correlation.
        """
        self.data = df
        self.target_variable = target_column
        self.factors = factors
        self.method = method
        self.threshold = threshold
        self.rolling_window = rolling_window
        self.valid_factors = []

    def compute_correlations(self):
        """
        Compute correlations between factors and the target variable.
        """
        correlations = {}
        for factor in self.factors:
            if self.rolling_window:
                # Compute rolling correlation
                rolling_corr = self.data[[factor, self.target_variable]].rolling(window=self.rolling_window).corr(method=self.method)
                # Extract the correlation values for the target variable
                correlations[factor] = rolling_corr.unstack()[self.target_variable]
            else:
                # Compute static correlation
                corr_value = self.data[[factor, self.target_variable]].corr(method=self.method).iloc[0, 1]
                correlations[factor] = corr_value
        self.correlations = correlations

    def filter_factors(self):
        """
        Filter factors based on the correlation threshold.
        """
        self.valid_factors = [factor for factor, corr in self.correlations.items() if abs(corr) >= self.threshold]

    def get_valid_factors(self):
        """
        Return the list of valid factors.
        """
        return self.valid_factors


# Example Usage
# Assuming `df` has columns: 'TRADE_DT', 'FUTURE_TICKER', target column (e.g., 'RETURN'), and factor columns
factors = ["FACTOR_1", "FACTOR_2", "FACTOR_3"]  # Replace with actual factor names
target_column = "RETURN"

tester = FactorCorrelationAnalyzer(correlation_threshold=0.02, rolling_window=30, method="pearson")
df_with_corr, valid_factors = tester.calculate_factor_correlation(df, factors, target_column)

print(f"Valid factors: {valid_factors}")
