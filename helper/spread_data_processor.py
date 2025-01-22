# spread_data_processor.py


from helper.future_price_retriever import FuturePriceRetriever


class SpreadDataProcessor:
    def __init__(self, spread_df):
        """
        Initializes the SpreadDataProcessor class to process spread data.

        Parameters:
        - spread_df: A DataFrame containing the spread data for the futures symbols.
        - window: The rolling window size for calculating mean and standard deviation (default is 20).
        """
        self.spread_df = spread_df

    def compute_moving_statistics(self, window: int):
        """
        Calculates the moving mean, standard deviation, and Z-score for the spread data.
        This method performs rolling window calculations on the RB_HC_spread column to compute:
        - Moving mean over specified window
        - Moving standard deviation over specified window
        - Z-score based on the moving mean and standard deviation
        Args:
            window (int): The size of the rolling window in days for calculating statistics
            pandas.DataFrame: DataFrame containing the original spread data plus new columns:
                - mean_{window}d: Moving average over specified window
                - sd_{window}d: Moving standard deviation over specified window
                - z_score_{window}d: Z-score calculated using the moving statistics
        Note:
            The method modifies the existing spread_df DataFrame by adding new columns.
            min_periods=1 is used in rolling calculations to start computing as soon as possible.
        """
        # Calculate moving mean and standard deviation
        self.spread_df[f"mean_{window}d"] = self.spread_df["RB_HC_spread"].rolling(window=window, min_periods=1).mean()
        self.spread_df[f"sd_{window}d"] = self.spread_df["RB_HC_spread"].rolling(window=window, min_periods=1).std()

        # Calculate Z-score
        self.spread_df["z_score_{window}d"] = (self.spread_df["RB_HC_spread"] - self.spread_df[f"mean_{window}d"]) / self.spread_df[f"sd_{window}d"]

        return self.spread_df


# Example usage
symbols = ["RB", "HC"]
future_price_retriever = FuturePriceRetriever()

# Retrieve spread data
spread_df = future_price_retriever.get_spread_data(symbols)

# Process the spread data
spread_processor = SpreadDataProcessor(spread_df)
processed_spread_df = spread_processor.calculate_statistics(window=20)

# Print the processed spread DataFrame
print(processed_spread_df.head())
