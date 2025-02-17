# pair_trade_strat.py


import numpy as np
import pandas as pd
from future_arb.future_arbitrage_strat import FutureArbitrageStrat
from helper.spread_data_processor import SpreadDataProcessor
from future_arb.signal_generator import SignalGenerator
from typing import Tuple


class PairTradeStrategy(SpreadDataProcessor, FutureArbitrageStrat, SignalGenerator):

    def __init__(self, trading_pair: Tuple[str, str]):
        """
        Initializes the PairTradeStrategy class.
        """
        SpreadDataProcessor.__init__(self)
        FutureArbitrageStrat.__init__(self)
        self.cost = self.slippage_risk + self.commission
        self.instrument_a = trading_pair[0]
        self.instrument_b = trading_pair[1]
        
    # def init_parameters(self, trading_pair: Tuple[str, str]):

    def calculate_binomial_pnl(
        self, spread_df, hedge_ratio: float, max_position_size: float = 100, position_a_col="POSITION_HC", position_b_col="POSITION_RB"
    ):
        """
        Calculates daily and cumulative PnL based on positions and price changes.

        Parameters:
        - spread_df: A DataFrame containing price and position data.
        - hedge_ratio: The ratio used to hedge RB against HC positions.
        - position_a_col: Column name for A positions.
        - position_b_col: Column name for B positions.

        Returns:
        - A DataFrame with daily and cumulative PnL calculated.
        """
        # Create a copy of the input DataFrame
        df = spread_df.copy()

        df["PNL"] = 0
        df["CUM_PNL"] = 0

        # Set position size based on signals
        # When BINOMIAL_SPREAD_SIGNAL = -1: short HC (-1), long RB (hedge_ratio)
        # When BINOMIAL_SPREAD_SIGNAL = 1: long HC (1), short RB (-hedge_ratio)
        # Shift the signal by one to avoid lookahead bias
        df[position_a_col] = df["BINOMIAL_SPREAD_SIGNAL"].shift(1) * max_position_size
        df[position_b_col] = -hedge_ratio * df["BINOMIAL_SPREAD_SIGNAL"].shift(1) * max_position_size
        # Fill first row NaN values with zeros
        df[position_b_col] = df[position_b_col].fillna(0)
        df[position_a_col] = df[position_a_col].fillna(0)
        for i in range(1, len(df)):
            prev_position_hc = df.iloc[i - 1][position_a_col]
            prev_position_rb = df.iloc[i - 1][position_b_col]
            curr_position_hc = df.iloc[i][position_a_col]
            curr_position_rb = df.iloc[i][position_b_col]

            # Calculate position change costs with cost
            position_change_hc = curr_position_hc - prev_position_hc
            position_change_rb = curr_position_rb - prev_position_rb
            slippage_cost = (
                abs(position_change_hc) * df.iloc[i]["HC_prices"] * self.cost + abs(position_change_rb) * df.iloc[i]["RB_prices"] * self.cost
            )

            # Calculate daily PnL
            daily_pnl = (
                prev_position_hc * (df.iloc[i]["HC_prices"] - df.iloc[i - 1]["HC_prices"])
                + prev_position_rb * (df.iloc[i]["RB_prices"] - df.iloc[i - 1]["RB_prices"])
                - slippage_cost
            )

            df.at[df.index[i], "PNL"] = daily_pnl
            df.at[df.index[i], "CUM_PNL"] = df.iloc[: i + 1]["PNL"].sum()

        return df

    def calculate_clipping_z_score_pnl(
        self,
        spread_df,
        hedge_ratio: float,
        entry_threshold: float,
        max_position_size: float,
        signal_col: str,
        window: int,
        position_hc_col="Z_SCORE_POSITION_HC",
        position_rb_col="Z_SCORE_POSITION_RB",
    ):
        """
        Calculates daily and cumulative PnL based on Z-score signals with risk management.

        Parameters:
        - spread_df: A DataFrame containing price and Z-score signal data
        - hedge_ratio: The ratio used to hedge RB against HC positions
        - entry_threshold: Z-score threshold for entering positions
        - max_position_size: Maximum position size as a percentage of trading capital
        - position_hc_col: Column name for HC positions
        - position_rb_col: Column name for RB positions

        Returns:
        - A DataFrame with positions and PnL calculations
        """
        df = spread_df.copy()
        df["PNL"] = 0
        df["CUM_PNL"] = 0

        # Normalize Z-score
        df["NORMALIZED_Z_SCORE"] = df[f"z_score_{window}d"] / entry_threshold
        df["NORMALIZED_Z_SCORE"] = df["NORMALIZED_Z_SCORE"].clip(-1, 1)

        # Calculate position sizes with one day lag
        # Shift the signal by one to avoid lookahead bias
        df[position_rb_col] = df["NORMALIZED_Z_SCORE"].shift(1) * max_position_size
        df[position_hc_col] = -hedge_ratio * df["NORMALIZED_Z_SCORE"].shift(1) * max_position_size
        # Fill first row NaN values with zeros
        df[position_rb_col] = df[position_rb_col].fillna(0)
        df[position_hc_col] = df[position_hc_col].fillna(0)

        for i in range(1, len(df)):
            prev_position_hc = df.iloc[i - 1][position_hc_col]
            prev_position_rb = df.iloc[i - 1][position_rb_col]
            curr_position_hc = df.iloc[i][position_hc_col]
            curr_position_rb = df.iloc[i][position_rb_col]

            # Calculate position change costs with cost
            position_change_hc = curr_position_hc - prev_position_hc
            position_change_rb = curr_position_rb - prev_position_rb
            slippage_cost = (
                abs(position_change_hc) * df.iloc[i]["HC_prices"] * self.cost + abs(position_change_rb) * df.iloc[i]["RB_prices"] * self.cost
            )

            # Calculate daily PnL
            daily_pnl = (
                prev_position_hc * (df.iloc[i]["HC_prices"] - df.iloc[i - 1]["HC_prices"])
                + prev_position_rb * (df.iloc[i]["RB_prices"] - df.iloc[i - 1]["RB_prices"])
                - slippage_cost
            )

            df.at[df.index[i], "PNL"] = daily_pnl
            df.at[df.index[i], "CUM_PNL"] = df.iloc[: i + 1]["PNL"].sum()

        return df

    def calculate_z_score_tanh_pnl(
        self,
        spread_df,
        hedge_ratio: float,
        entry_threshold: float,
        exit_threshold: float,
        max_position_size: float,
        min_holding_period: int,
        signal_col: str,
        window: int,
        sensitivity_factor: float,
        position_hc_col="Z_SCORE_POSITION_HC",
        position_rb_col="Z_SCORE_POSITION_RB",
    ):
        """
        Calculates daily and cumulative PnL based on Z-score signals with risk management.

        Parameters:
        - spread_df: A DataFrame containing price and Z-score signal data
        - hedge_ratio: The ratio used to hedge RB against HC positions
        - entry_threshold: Z-score threshold for entering positions
        - max_position_size: Maximum position size as a percentage of trading capital
        - position_hc_col: Column name for HC positions
        - position_rb_col: Column name for RB positions

        Returns:
        - A DataFrame with positions and PnL calculations
        """
        df = spread_df.copy()
        df["PNL"] = 0
        df["CUM_PNL"] = 0

        # Normalize Z-score
        df["TANH_Z_SIGNAL"] = np.tanh(sensitivity_factor * df[f"z_score_{window}d"])
        df["TANH_Z_SIGNAL"] = df["TANH_Z_SIGNAL"].shift(1)
        df["position_size"] = df["TANH_Z_SIGNAL"] * max_position_size

        # Apply entry threshold (only take trades when signal is strong)
        df["position_size"] = np.where(abs(df["TANH_Z_SIGNAL"]) > entry_threshold, df["position_size"], 0)
        # Apply exit threshold (hold trades longer to avoid frequent flipping)
        df["position_size"] = np.where((abs(df["TANH_Z_SIGNAL"]) < exit_threshold), 0, df["position_size"])

        # Track position holding time
        df["holding_days"] = df["position_size"].diff().ne(0).cumsum()
        df["position_size"] = np.where(df["holding_days"] < min_holding_period, df["position_size"].shift(1), df["position_size"])

        # Calculate position sizes with one day lag
        # Shift the signal by one to avoid lookahead bias
        df[position_rb_col] = df["position_size"]
        df[position_hc_col] = -hedge_ratio * df["position_size"]
        # Fill first row NaN values with zeros
        df[position_rb_col] = df[position_rb_col].fillna(0)
        df[position_hc_col] = df[position_hc_col].fillna(0)

        for i in range(1, len(df)):
            prev_position_hc = df.iloc[i - 1][position_hc_col]
            prev_position_rb = df.iloc[i - 1][position_rb_col]
            curr_position_hc = df.iloc[i][position_hc_col]
            curr_position_rb = df.iloc[i][position_rb_col]

            # Calculate position change costs with cost
            position_change_hc = curr_position_hc - prev_position_hc
            position_change_rb = curr_position_rb - prev_position_rb
            slippage_cost = (
                abs(position_change_hc) * df.iloc[i]["HC_prices"] * self.cost + abs(position_change_rb) * df.iloc[i]["RB_prices"] * self.cost
            )

            # Calculate daily PnL
            daily_pnl = (
                prev_position_hc * (df.iloc[i]["HC_prices"] - df.iloc[i - 1]["HC_prices"])
                + prev_position_rb * (df.iloc[i]["RB_prices"] - df.iloc[i - 1]["RB_prices"])
                - slippage_cost
            )

            df.at[df.index[i], "PNL"] = daily_pnl
            df.at[df.index[i], "CUM_PNL"] = df.iloc[: i + 1]["PNL"].sum()

        return df

    def normalize_signals_to_positions(
        self,
        signal_df,
        signal_col,
        max_position_size,
        hedge_ratio,
        signal_sensitivity: float,
        entry_threshold=None,
        exit_threshold=None,
        min_holding_period=None,
    ):
        """
        Normalizes signals and converts them to daily positions.

        Parameters:
        - df: DataFrame containing price and signal data
        - signal_col: Column name for the signal
        - max_position_size: Maximum position size as a percentage of trading capital
        - hedge_ratio: The ratio used to hedge RB against HC positions
        - signal_sensitivity: Parameter to adjust the sensitivity of the tanh function
        - entry_threshold: Z-score threshold for entering positions (optional)
        - exit_threshold: Z-score threshold for exiting positions (optional)
        - min_holding_period: Minimum holding period for positions (optional)

        Returns:
        - DataFrame with position sizes
        """
        df = signal_df.copy()
        df["TEMPORAL_TANH_Z_SIGNAL"] = np.tanh(signal_sensitivity * df[signal_col])
        df["TEMPORAL_TANH_Z_SIGNAL"] = df["TEMPORAL_TANH_Z_SIGNAL"].shift(1)
        df["position_size"] = df["TEMPORAL_TANH_Z_SIGNAL"] * max_position_size

        if entry_threshold is not None:
            df["position_size"] = np.where(abs(df["TEMPORAL_TANH_Z_SIGNAL"]) > entry_threshold, df["position_size"], 0)
        if exit_threshold is not None:
            df["position_size"] = np.where((abs(df["TEMPORAL_TANH_Z_SIGNAL"]) < exit_threshold), 0, df["position_size"])
        if min_holding_period is not None:
            df["holding_days"] = df["position_size"].diff().ne(0).cumsum()
            df["position_size"] = np.where(df["holding_days"] < min_holding_period, df["position_size"].shift(1), df["position_size"])

        position_rb_col = f"{signal_col}_POSITION_{self.instrument_a}"
        position_hc_col = f"{signal_col}_POSITION_{self.instrument_b}"

        df[position_rb_col] = df["position_size"]
        df[position_hc_col] = -hedge_ratio * df["position_size"]
        df[position_rb_col] = df[position_rb_col].fillna(0)
        df[position_hc_col] = df[position_hc_col].fillna(0)

        df.drop(columns=["TEMPORAL_TANH_Z_SIGNAL"], inplace=True)

        return df

    def calculate_pnl_and_sharpe_ratio(self, df, position_hc_col, position_rb_col):
        """
        Calculates daily and cumulative PnL, cost, and Sharpe ratio.

        Parameters:
        - df: DataFrame containing price and position data
        - position_hc_col: Column name for HC positions
        - position_rb_col: Column name for RB positions

        Returns:
        - DataFrame with PnL calculations
        """
        df["PNL"] = 0
        df["CUM_PNL"] = 0

        for i in range(1, len(df)):
            prev_position_hc = df.iloc[i - 1][position_hc_col]
            prev_position_rb = df.iloc[i - 1][position_rb_col]
            curr_position_hc = df.iloc[i][position_hc_col]
            curr_position_rb = df.iloc[i][position_rb_col]

            position_change_hc = curr_position_hc - prev_position_hc
            position_change_rb = curr_position_rb - prev_position_rb
            slippage_cost = (
                abs(position_change_hc) * df.iloc[i]["HC_prices"] * self.cost + abs(position_change_rb) * df.iloc[i]["RB_prices"] * self.cost
            )

            daily_pnl = (
                prev_position_hc * (df.iloc[i]["HC_prices"] - df.iloc[i - 1]["HC_prices"])
                + prev_position_rb * (df.iloc[i]["RB_prices"] - df.iloc[i - 1]["RB_prices"])
                - slippage_cost
            )

            df.at[df.index[i], "PNL"] = daily_pnl
            df.at[df.index[i], "CUM_PNL"] = df.iloc[: i + 1]["PNL"].sum()

        daily_returns = df["PNL"]
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Assuming 252 trading days in a year
        df["SHARPE_RATIO"] = sharpe_ratio

        return df

    def calculate_z_score_pnl(
        self,
        spread_df,
        hedge_ratio: float,
        entry_threshold: float,
        exit_threshold: float,
        max_position_size: float,
        min_holding_period: int,
        signal_col: str,
        window: int,
        position_hc_col="Z_SCORE_POSITION_HC",
        position_rb_col="Z_SCORE_POSITION_RB",
    ):
        """
        Calculates daily and cumulative PnL based on Z-score signals with risk management.

        Parameters:
        - spread_df: A DataFrame containing price and Z-score signal data
        - hedge_ratio: The ratio used to hedge RB against HC positions
        - entry_threshold: Z-score threshold for entering positions
        - max_position_size: Maximum position size as a percentage of trading capital
        - position_hc_col: Column name for HC positions
        - position_rb_col: Column name for RB positions

        Returns:
        - A DataFrame with positions and PnL calculations
        """
        df = spread_df.copy()
        df = self.normalize_signals_to_positions(
            df, f"z_score_{window}d", max_position_size, hedge_ratio, entry_threshold, exit_threshold, min_holding_period
        )
        df = self.calculate_pnl_and_sharpe_ratio(df, position_hc_col, position_rb_col)

        return df
