# factor_model.py


import numpy as np
import pandas as pd
from common.log import Log

Log = Log()


class BaseFactorModel:
    def __init__(self):
        self.df = pd.DataFrame()

    def get_df(self):
        return self.df.copy()

    def calculate_factor(
        self,
        df: pd.DataFrame,
        target_column: str,
        group_column: str = "FS_INFO_SCCODE",
        factor_types: list[str] = ["SKEW", "KURTOSIS", "VOLATILITY", "MEAN", "ROC", "PAMR", "EMA"],
        periods: list[int] = [7, 14, 30, 60],
    ) -> pd.DataFrame:
        """
        统一的因子计算方法，根据factor_type计算不同类型的因子

        :param df: Input DataFrame with required columns
        :param factor_type: Type of factor to calculate ('skew', 'kurtosis', 'volatility', 'mean')
        :param target_column: The column name for which factor is calculated
        :param periods: A list of periods over which to calculate the factor
        :return: pd.DataFrame with calculated factors added as new columns
        """
        # Sort only by TRADE_DT within each group
        # Sanity-check. If sorted before, then it runs in O(N)
        df.sort_values(by="TRADE_DT", inplace=True)

        VALID_FACTORS = ["SKEW", "KURTOSIS", "VOLATILITY", "MEAN", "ROC", "PAMR", "EMA"]
        # Convert all factor types to uppercase and validate
        factor_types = [ft.upper() for ft in factor_types]
        invalid_factors = [ft for ft in factor_types if ft not in VALID_FACTORS]
        if invalid_factors:
            raise ValueError(f"Invalid factor type(s): {invalid_factors}. Must be one of {VALID_FACTORS}")

        min_periods = 1
        for factor_type in factor_types:
            for period in periods:
                col_name = f"{factor_type}_{target_column}_{period}_FACTOR"

                # Group by FUTURE_TICKER and calculate rolling statistics
                grouped = df.groupby(group_column)[target_column]

                if factor_type == "SKEW":
                    df[col_name] = grouped.rolling(window=period, min_periods=min_periods).skew().reset_index(level=0, drop=True)
                elif factor_type == "KURTOSIS":
                    df[col_name] = grouped.rolling(window=period, min_periods=min_periods).kurt().reset_index(level=0, drop=True)
                elif factor_type == "VOLATILITY":
                    df[col_name] = grouped.rolling(window=period, min_periods=min_periods).std().reset_index(level=0, drop=True)
                elif factor_type == "MEAN":
                    df[col_name] = grouped.rolling(window=period, min_periods=min_periods).mean().reset_index(level=0, drop=True)
                elif factor_type == "ROC":
                    df[col_name] = grouped.pct_change(periods=period).reset_index(level=0, drop=True)
                elif factor_type == "PAMR":
                    # Calculate Price Moving Average Ratio factor: price / mean(price, n)
                    rolling_mean = grouped.rolling(window=period, min_periods=min_periods).mean().reset_index(level=0, drop=True)
                    df[col_name] = df[target_column] / rolling_mean
                elif factor_type == "EMA":
                    df[col_name] = grouped.transform(lambda x: x.ewm(span=period, adjust=False).mean()).reset_index(level=0, drop=True)

        return df

    def calculate_rsi_factor(
        self, df: pd.DataFrame, group_column: str = "FS_INFO_SCCODE", target_column: str = "S_DQ_SETTLE", periods: list[int] = [7, 14]
    ) -> pd.DataFrame:
        """
        Calculate the RSI (Relative Strength Index) factor to measure overbought or oversold conditions.

        Formula:
            RSI = 100 - (100 / (1 + RS))
            RS = Average Gain over the period / Average Loss over the period

        Columns used:
            - target_column: The column for which RSI is calculated (default is "S_DQ_SETTLE").
            - period: The number of days to calculate the RSI (default is 14).

        :param df: Input DataFrame with required columns
        :param target_column: The column name for the price data (default is "S_DQ_CLOSE").
        :param period: Lookback period for RSI calculation (default is 14).
        :return: pd.DataFrame with RSI factor added as a new column.
        """
        # Validate required columns
        if target_column not in df.columns:
            raise ValueError(f"Required column '{target_column}' is missing from the input DataFrame.")

        # Sort only by TRADE_DT within each group
        # Sanity-check. If sorted before, then it runs in O(N)
        df.sort_values(by="TRADE_DT", inplace=True)

        # Calculate daily price changes within each FUTURE_TICKER group
        df["PRICE_CHANGE"] = df.groupby(group_column)[target_column].diff(periods=1)

        # Separate gains and losses
        df["GAIN"] = df["PRICE_CHANGE"].clip(lower=0, upper=None)
        df["LOSS"] = -df["PRICE_CHANGE"].clip(upper=0)

        min_period = 1
        for period in periods:
            col_name = f"RSI_{period}_FACTOR"
            # Calculate the rolling average of gains and losses
            df["AVG_GAIN"] = df.groupby("FUTURE_TICKER")["GAIN"].rolling(window=period, min_periods=min_period).mean().reset_index(level=0, drop=True)
            df["AVG_LOSS"] = df.groupby("FUTURE_TICKER")["LOSS"].rolling(window=period, min_periods=min_period).mean().reset_index(level=0, drop=True)

            # Calculate RS (Relative Strength)
            df["RS"] = df["AVG_GAIN"] / df["AVG_LOSS"]

            # Calculate RSI (Relative Strength Index)
            df[col_name] = 100 - (100 / (1 + df["RS"]))

        # Clean up intermediate columns
        df.drop(columns=["PRICE_CHANGE", "GAIN", "LOSS", "AVG_GAIN", "AVG_LOSS", "RS"], inplace=True)

        return df

    def calculate_fast_slow_ema_factor(
        self, df: pd.DataFrame, target_column: str = "S_DQ_SETTLE", fast_period: int = 7, slow_period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate the fast-slow EMA factor for a given price column.

        Formula:
            SEMA = EMA(price, slow_period)
            FEMA = EMA(price, fast_period)
            factor = (FEMA - SEMA) / SEMA

        :param df: Input DataFrame with required columns
        :param target_column: The column name for the price data (default is "S_DQ_SETTLE").
        :param fast_period: Period for the fast EMA (default is 7).
        :param slow_period: Period for the slow EMA (default is 14).
        :return: pd.DataFrame with the calculated fast-slow EMA factor added as a new column.
        """
        # Check for required columns
        ema_columns = [f"EMA_{target_column}_{fast_period}_FACTOR", f"EMA_{target_column}_{slow_period}_FACTOR"]
        missing = [col for col in ema_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Required columns missing: {missing}")

        # Calculate factor directly without intermediate columns
        df["FAST_SLOW_EMA_FACTOR"] = (df[f"EMA_{target_column}_{fast_period}_FACTOR"] - df[f"EMA_{target_column}_{slow_period}_FACTOR"]) / df[
            f"EMA_{target_column}_{slow_period}_FACTOR"
        ]

        return df

    def calculate_mid_price_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        中间价因子

        Formula:
            factor = (close + (high + low) / 2) / settlement

        Columns used:
            - S_DQ_HIGH: High price
            - S_DQ_LOW: Low price
            - S_DQ_CLOSE: Close price
            - S_DQ_SETTLE: Settlement price

        :param df: Input DataFrame with required columns
        :return: pd.DataFrame with Mid-Price Factor added as a new column
        """
        # Validate required columns are present
        required_columns = ["S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE", "S_DQ_SETTLE"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' is missing from the input DataFrame.")

        # Calculate Mid-Price Factor
        df["MID_PRICE_FACTOR"] = (df["S_DQ_CLOSE"] + (df["S_DQ_HIGH"] + df["S_DQ_LOW"]) / 2) / df["S_DQ_SETTLE"]
        df.loc[df["S_DQ_SETTLE"] == 0, "MID_PRICE_FACTOR"] = float("nan")

        return df

    def calculate_max_factor(
        self, df: pd.DataFrame, group_column: str = "FS_INFO_SCCODE", target_column: str = "S_DQ_SETTLE", lookback_period: int = 14, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Calculate the MAX factor for a given price column.

        Formula:
            r_i = (P_t - P_{t-1}) / P_{t-1}
            s = \\prod^N_{i \in maxprod}(1 + r_i) for the top N days with the highest returns in the past lookback_period days.

        :param df: Input DataFrame with required columns.
        :param target_column: The column name for the price data (default is "S_DQ_SETTLE").
        :param lookback_period: The number of days to look back for calculating the factor (default is 14).
        :param top_n: The number of top days with the highest returns to consider in the calculation (default is 10).
        :return: pd.DataFrame with the calculated MAX factor added as a new column.
        """
        # Validate required columns
        if target_column not in df.columns:
            raise ValueError(f"Required column '{target_column}' is missing from the input DataFrame.")

        # Sort by FUTURE_TICKER and TRADE_DT
        df.sort_values(by=["TRADE_DT"], inplace=True)

        # Calculate daily returns
        df["TEMP_DAILY_RETURN"] = df.groupby(group_column)[target_column].pct_change()

        # Calculate MAX factor using rolling window and top N returns
        df["MAX_FACTOR"] = (
            df.groupby(group_column)["TEMP_DAILY_RETURN"]
            .rolling(window=lookback_period, min_periods=1)
            .apply(lambda x: (1 + x.nlargest(min(top_n, len(x)))).prod())
            .reset_index(level=0, drop=True)
        )

        # Clean up intermediate columns
        df.drop(columns=["TEMP_DAILY_RETURN"], inplace=True)

        return df

    def single_factor_test(
        self,
        df,
        factor_column,
        group_column: str = "FS_INFO_SCCODE",
        target_column: str = "S_DQ_SETTLE",
        rolling_window=30,
        correlation_threshold=0.02,
        method="pearson",
    ) -> pd.DataFrame:
        """
        Perform single-factor testing by calculating the correlation between a factor and the target variable.
        Factors with rolling correlation below a specified threshold are excluded.

        :param factor_column: The column name of the factor to be tested.
        :param target_column: The column name of the target variable (e.g., returns).
        :param rolling_window: The window size for calculating rolling correlations.
        :param correlation_threshold: The minimum acceptable correlation for the factor to be retained.
        :param method: Correlation method, either 'pearson' or 'spearman'.
        :return: pd.DataFrame with rolling correlations and a mask for valid factors.
        """
        # Ensure the DataFrame is sorted by date for rolling calculations
        df = df.sort_values(by=["FUTURE_TICKER", "TRADE_DT"])

        # Calculate rolling correlation grouped by FUTURE_TICKER
        df["ROLLING_CORRELATION"] = (
            df.groupby(group_column)
            .apply(lambda x: x[[factor_column, target_column]].rolling(window=rolling_window).corr(method=method).unstack().iloc[:, 1])
            .reset_index(level=0, drop=True)
        )

        # Mask factors with correlation below the threshold
        df["IS_VALID_FACTOR"] = df["ROLLING_CORRELATION"] >= correlation_threshold

        # Filter the DataFrame to retain only valid factors
        valid_factors_df = df[df["IS_VALID_FACTOR"]].copy()

        return valid_factors_df.sort_index()
