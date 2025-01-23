# spread_data_processor.py


import pandas as pd


class SpreadDataProcessor:
    def __init__(self):
        """
        Initializes the SpreadDataProcessor class to process spread data.

        Parameters:
        - spread_df: A DataFrame containing the spread data for the futures symbols.
        - window: The rolling window size for calculating mean and standard deviation (default is 20).
        """
        pass

    def compute_moving_statistics(self, df: pd.DataFrame, window: int | list[int]):
        """
        Calculates the moving mean, standard deviation, and Z-score for the spread data.
        This method performs rolling window calculations on the RB_HC_spread column to compute:
        - Moving mean over specified window(s)
        - Moving standard deviation over specified window(s)
        - Z-score based on the moving mean and standard deviation
        Args:
            window (int | list[int]): The size(s) of the rolling window(s) in days for calculating statistics
            pandas.DataFrame: DataFrame containing the original spread data plus new columns:
                - mean_{window}d: Moving average over specified window(s)
                - sd_{window}d: Moving standard deviation over specified window(s)
                - z_score_{window}d: Z-score calculated using the moving statistics
        Note:
            The method modifies the existing spread_df DataFrame by adding new columns.
            min_periods=1 is used in rolling calculations to start computing as soon as possible.
        """
        spread_df = df.copy()
        if isinstance(window, int):
            window = [window]

        for w in window:
            # Calculate moving mean and standard deviation
            spread_df[f"mean_{w}d"] = spread_df["RB_HC_spread"].rolling(window=w, min_periods=1).mean()
            spread_df[f"sd_{w}d"] = spread_df["RB_HC_spread"].rolling(window=w, min_periods=1).std()

            # Calculate Z-score
            spread_df[f"z_score_{w}d"] = (spread_df["RB_HC_spread"] - spread_df[f"mean_{w}d"]) / spread_df[f"sd_{w}d"]

        return spread_df
