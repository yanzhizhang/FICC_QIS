# mean_reversion.py


from helper.spread_data_processor import SpreadDataProcessor


class SpreadTradingStrategy(SpreadDataProcessor):
    def __init__(self):
        """
        Initializes the SpreadTradingStrategy class.
        """
        pass

    def evaluate_binomial_signals(self, spread_df, window: int, entry_threshold: float, exit_threshold: float):
        """
        Generates trading signals based on Z-score thresholds.

        Parameters:
        - spread_df: A DataFrame containing spread data and Z-scores.

        Returns:
        - A DataFrame with generated signals for each day.
        """
        df = spread_df.copy()
        df["BINOMIAL_SPREAD_SIGNAL"] = 0

        for i in range(len(df)):
            if df.iloc[i][f"z_score_{window}d"] < -entry_threshold:
                # Spread is too large: short HC, long RB, BINOMIAL_SPREAD_SIGNAL = 1
                df.at[df.index[i], "BINOMIAL_SPREAD_SIGNAL"] = 1
            elif df.iloc[i][f"z_score_{window}d"] > entry_threshold:
                # Spread is too small: long HC, short RB, BINOMIAL_SPREAD_SIGNAL = -1
                df.at[df.index[i], "BINOMIAL_SPREAD_SIGNAL"] = -1
            # Uncomment to handle exit threshold logic:
            elif abs(df.iloc[i][f"z_score_{window}d"]) <= exit_threshold:
                # Spread is about average: exit position, BINOMIAL_SPREAD_SIGNAL = -1
                df.at[df.index[i], "BINOMIAL_SPREAD_SIGNAL"] = 0

        return df

    def evaluate_z_score_signals(self, spread_df, window: int, signal_col: str):
        """
        Generates continuous trading signals based on Z-score deviations.

        Parameters:
        - spread_df: A DataFrame containing spread data and Z-scores.
        - window: The rolling window size used for Z-score calculation.

        Returns:
        - A DataFrame with generated continuous signals for each day.
        """
        df = spread_df.copy()
        z_score_col = f"z_score_{window}d"

        # Ensure the Z-score column exists
        if z_score_col not in df.columns:
            raise ValueError(f"The DataFrame must contain a '{z_score_col}' column.")

        # Generate continuous signals proportional to the Z-score
        df[signal_col] = -df[z_score_col]

        return df

    def calculate_binomial_pnl(
        self, spread_df, hedge_ratio: float, max_position_size: float = 10, position_hc_col="POSITION_HC", position_rb_col="POSITION_RB"
    ):
        """
        Calculates daily and cumulative PnL based on positions and price changes.

        Parameters:
        - spread_df: A DataFrame containing price and position data.
        - hedge_ratio: The ratio used to hedge RB against HC positions.
        - position_hc_col: Column name for HC positions.
        - position_rb_col: Column name for RB positions.

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
        df[position_hc_col] = df["BINOMIAL_SPREAD_SIGNAL"].shift(1) * max_position_size
        df[position_rb_col] = -hedge_ratio * df["BINOMIAL_SPREAD_SIGNAL"].shift(1) * max_position_size
        # Fill first row NaN values with zeros
        df[position_rb_col] = df[position_rb_col].fillna(0)
        df[position_hc_col] = df[position_hc_col].fillna(0)

        for i in range(1, len(df)):
            prev_position_hc = df.iloc[i - 1][position_hc_col]
            prev_position_rb = df.iloc[i - 1][position_rb_col]

            # Calculate daily PnL
            daily_pnl = prev_position_hc * (df.iloc[i]["HC_prices"] - df.iloc[i - 1]["HC_prices"]) + prev_position_rb * (
                df.iloc[i]["RB_prices"] - df.iloc[i - 1]["RB_prices"]
            )

            df.at[df.index[i], "PNL"] = daily_pnl
            df.at[df.index[i], "CUM_PNL"] = df.iloc[: i + 1]["PNL"].sum()

        return df

    def calculate_z_score_pnl(
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

            # Calculate daily PnL
            daily_pnl = prev_position_hc * (df.iloc[i]["HC_prices"] - df.iloc[i - 1]["HC_prices"]) + prev_position_rb * (
                df.iloc[i]["RB_prices"] - df.iloc[i - 1]["RB_prices"]
            )

            df.at[df.index[i], "PNL"] = daily_pnl
            df.at[df.index[i], "CUM_PNL"] = df.iloc[: i + 1]["PNL"].sum()

        return df
