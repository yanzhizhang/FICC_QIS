# lightgbm_strategy.py

import lightgbm as lgb
import numpy as np
import pandas as pd
from future_arb.future_arbitrage_strat import FutureArbitrageStrat
from future_arb.loss_function import LightGbmLossFunction
from sklearn.model_selection import train_test_split


class LightGBMSpreadTradingStrategy(LightGbmLossFunction, FutureArbitrageStrat):
    def __init__(self, hedge_ratio: float):
        """
        Initializes the LightGBMSpreadTradingStrategy class.
        """
        FutureArbitrageStrat().__init__()
        self.model = None
        self.hedge_ratio = hedge_ratio

    def train_model(self, spread_df: pd.DataFrame, feature_cols: list[str], target_col: list[str]):
        """
        Trains a LightGBM model to predict trading signals.

        Parameters:
        - spread_df: A DataFrame containing spread data and Z-scores.
        - feature_cols: A list of column names for features.
        - target_col: The column name for the target variable (e.g., future price change).
        """
        X = spread_df[feature_cols]
        y = spread_df[target_col]

        train_data = lgb.Dataset(X, label=y)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        self.model = lgb.train(params, train_data, num_boost_round=100)

    def predict_signals(self, spread_df: pd.DataFrame, feature_cols: list[str], signal_col: str):
        """
        Uses the trained LightGBM model to predict trading signals.

        Parameters:
        - spread_df: A DataFrame containing spread data and Z-scores.
        - feature_cols: A list of column names for features.
        - signal_col: The column name for the predicted signals.

        Returns:
        - A DataFrame with the predicted signals.
        """
        if self.model is None:
            raise ValueError("The model must be trained before making predictions.")

        X = spread_df[feature_cols]
        spread_df[signal_col] = self.model.predict(X)

        return spread_df

    def calculate_position_size(y_pred, max_position_size, threshold, hedge_ratio):
        """
        Calculates the position size for RB and HC based on y_pred.

        Parameters:
        - y_pred: Predicted spread (RB - HC).
        - max_position_size: Maximum allowable position size.
        - threshold: Minimum spread magnitude to trigger a trade.
        - hedge_ratio: Ratio to hedge RB against HC.

        Returns:
        - (position_rb, position_hc): Long/short positions for RB and HC.
        """
        if abs(y_pred) < threshold:
            # No position if predicted spread is below threshold
            return 0, 0

        # Scale position size based on signal strength
        position_size = min(max_position_size, (abs(y_pred) / threshold) * max_position_size)

        if y_pred > 0:
            # Long spread: Buy RB, Sell HC
            position_rb = position_size
            position_hc = -position_size * hedge_ratio
        else:
            # Short spread: Sell RB, Buy HC
            position_rb = -position_size
            position_hc = position_size * hedge_ratio

        return position_rb, position_hc

    def calculate_pnl(
        self,
        spread_df: pd.DataFrame,
        hedge_ratio: float,
        max_position_size: float,
        signal_col: str,
        position_hc_col="LGBM_POSITION_HC",
        position_rb_col="LGBM_POSITION_RB",
    ):
        """
        Calculates daily and cumulative PnL based on LightGBM signals.

        Parameters:
        - spread_df: A DataFrame containing price and signal data.
        - hedge_ratio: The ratio used to hedge RB against HC positions.
        - max_position_size: Maximum position size as a percentage of trading capital.
        - signal_col: The column name for the predicted signals.
        - position_hc_col: Column name for HC positions.
        - position_rb_col: Column name for RB positions.

        Returns:
        - A DataFrame with positions and PnL calculations.
        """
        df = spread_df.copy()
        df["PNL"] = 0
        df["CUM_PNL"] = 0

        # Calculate position sizes
        df[position_rb_col] = df[signal_col] * max_position_size
        df[position_hc_col] = -hedge_ratio * df[signal_col] * max_position_size
        df[position_rb_col] = df[position_rb_col].fillna(0)
        df[position_hc_col] = df[position_hc_col].fillna(0)

        for i in range(1, len(df)):
            prev_position_hc = df.iloc[i - 1][position_hc_col]
            prev_position_rb = df.iloc[i - 1][position_rb_col]

            daily_pnl = prev_position_hc * (df.iloc[i]["HC_prices"] - df.iloc[i - 1]["HC_prices"]) + prev_position_rb * (
                df.iloc[i]["RB_prices"] - df.iloc[i - 1]["RB_prices"]
            )

            df.at[df.index[i], "PNL"] = daily_pnl
            df.at[df.index[i], "CUM_PNL"] = df.iloc[: i + 1]["PNL"].sum()

        return df

    def calculate_spread_change_signal_pnl(
        self,
        spread_df: pd.DataFrame,
        hedge_ratio: float,
        max_position_size: float,
        signal_col: str = "LIGHTGBM_SPREAD_CHANGE_SIGNAL",
        position_hc_col="LGBM_POSITION_HC",
        position_rb_col="LGBM_POSITION_RB",
    ):
        """
        Calculates daily and cumulative PnL based on LightGBM signals.

        Parameters:
        - spread_df: A DataFrame containing price and signal data.
        - hedge_ratio: The ratio used to hedge RB against HC positions.
        - max_position_size: Maximum position size as a percentage of trading capital.
        - signal_col: The column name for the predicted signals.
        - position_hc_col: Column name for HC positions.
        - position_rb_col: Column name for RB positions.

        Returns:
        - A DataFrame with positions and PnL calculations.
        """
        df = spread_df.copy()
        df["PNL"] = 0
        df["CUM_PNL"] = 0

        # Calculate position sizes
        if df[signal_col].abs() <= 1:
            df[position_rb_col] = 0
            df[position_hc_col] = 0
        elif df[signal_col] > 2:
            df[position_rb_col] = - max_position_size
            df[position_hc_col] = hedge_ratio * max_position_size
        elif df[signal_col] < -2:
            df[position_rb_col] = max_position_size
            df[position_hc_col] = - hedge_ratio * max_position_size

        for i in range(1, len(df)):
            prev_position_hc = df.iloc[i - 1][position_hc_col]
            prev_position_rb = df.iloc[i - 1][position_rb_col]

            daily_pnl = prev_position_hc * (df.iloc[i]["HC_prices"] - df.iloc[i - 1]["HC_prices"]) + prev_position_rb * (
                df.iloc[i]["RB_prices"] - df.iloc[i - 1]["RB_prices"]
            )

            df.at[df.index[i], "PNL"] = daily_pnl
            df.at[df.index[i], "CUM_PNL"] = df.iloc[: i + 1]["PNL"].sum()

        return df

    def split_data(self, spread_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Splits the dataset into training and testing sets.

        Parameters:
        - spread_df: A DataFrame containing the spread data.
        - test_size: The proportion of the dataset to include in the test split.
        - random_state: The seed used by the random number generator.

        Returns:
        - A tuple containing the training and testing DataFrames.
        """
        train_df, test_df = train_test_split(spread_df, test_size=test_size, random_state=random_state)
        return train_df, test_df
