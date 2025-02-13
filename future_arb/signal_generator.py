# signal_generator.py

import pandas as pd


class SignalGenerator:
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
