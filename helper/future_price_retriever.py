import datetime as dt

import pandas as pd
import pytz
import rqdatac


class FuturePriceRetriever:
    def __init__(self, start_date="20150101", end_date=None):
        """
        Initializes the FuturePriceRetriever class with start and end dates.

        Parameters:
        - start_date: The start date for data retrieval in YYYYMMDD format (default is "20150101").
        - end_date: The end date for data retrieval in YYYYMMDD format (default is today's date).
        """
        self.start_date = start_date
        self.end_date = end_date if end_date else dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d")
        rqdatac.init()

    def get_future_price_data(self, symbols, frequency="1d", fields=None, adjust_type="pre", adjust_method="prev_close_spread"):
        """
        Retrieves the dominant future price data for given symbols from a specified time range.

        Parameters:
        - symbols: A list of future symbols to retrieve data for (e.g., ['RB', 'HC']).

        Returns:
        - A DataFrame containing the price data of the given symbols.
        """
        price_data = {}
        for symbol in symbols:
            price_df = rqdatac.futures.get_dominant_price(
                underlying_symbols=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                frequency=frequency,
                fields=fields,
                adjust_type=adjust_type,
                adjust_method=adjust_method,
            )
            prices = (price_df["open"] + price_df["close"] + price_df["high"] + price_df["low"]) / 4
            price_data[symbol] = prices.droplevel(level=0)

        price_df = pd.DataFrame(price_data)
        return price_df.copy()

    def get_spread_data(self, symbols):
        """
        Calculates the spread between two symbols (e.g., RB and HC) and returns the spread data.

        Parameters:
        - symbols: A list of two future symbols (e.g., ['RB', 'HC']).

        Returns:
        - A DataFrame containing the spread between the given symbols.
        """
        price_df = self.get_future_price_data(symbols)
        if len(symbols) == 2:
            spread_df = pd.DataFrame(
                {
                    f"{symbols[1]}_prices": price_df[symbols[1]],
                    f"{symbols[0]}_prices": price_df[symbols[0]],
                    f"{symbols[0]}_{symbols[1]}_spread": price_df[symbols[0]] - price_df[symbols[1]],
                }
            )
            return spread_df.copy()
        else:
            raise ValueError("The 'symbols' list must contain exactly two symbols to calculate spread.")


# # Example usage
# symbols = ["RB", "HC"]
# future_price_retriever = FuturePriceRetriever()

# # Retrieve price data
# price_df = future_price_retriever.get_future_price_data(symbols)
# print(price_df.head())

# """
#                RB    HC
# date
# 2015-01-05   62.0  40.0
# 2015-01-06   92.0  34.0
# 2015-01-07  107.0  41.0
# 2015-01-08   89.0  15.0
# 2015-01-09   77.5 -17.0
# """

# # Retrieve spread data
# spread_df = future_price_retriever.get_spread_data(symbols)
# print(spread_df.head())

# """
#             HC_prices  RB_prices  RB-HC
# date
# 2015-01-05       40.0       62.0   22.0
# 2015-01-06       34.0       92.0   58.0
# 2015-01-07       41.0      107.0   66.0
# 2015-01-08       15.0       89.0   74.0
# 2015-01-09      -17.0       77.5   94.5
# """
