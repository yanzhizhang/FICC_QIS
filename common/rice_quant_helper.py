import pandas as pd
import rqdatac
from common.log import Log

Log = Log()


# This class likely provides helper functions or methods related to quantitative analysis of rice.
class RiceQuantHelper:
    def __init__(self):
        """Initialize the MkApi class by setting up the rqdatac connection."""
        rqdatac.init()

    def get_instrument_info(self, contract_type=None, date=None):
        """
        Retrieve basic contract information from rqdatac.

        Parameters:
        - contract_type (str): Type of the contract (e.g., "Future").
        - date (str): Specific date to filter tradable contracts.

        Returns:
        - pd.DataFrame: A DataFrame containing contract details, including:
          ['order_book_id', 'underlying_symbol', 'market_tplus', 'symbol',
           'margin_rate', 'maturity_date', 'type', 'trading_code', 'exchange',
           'product', 'contract_multiplier', 'round_lot', 'trading_hours',
           'listed_date', 'industry_name', 'de_listed_date',
           'underlying_order_book_id', 'start_delivery_date', 'end_delivery_date']
        """
        contract_info = rqdatac.all_instruments(contract_type, date)
        return contract_info if contract_info is not None else pd.DataFrame()

    def get_price_data(
        self, order_book_ids: str | list[str], start_date: str, end_date: str, frequency="1d", time_slice=None, expect_df=True, adjust_type="none"
    ):
        """
        Retrieve price data for specific contracts.

        Parameters:
        - order_book_ids (list or str): List of order book IDs or a single ID.
        - start_date (str): Start date for price data (YYYY-MM-DD).
        - end_date (str): End date for price data (YYYY-MM-DD).
        - frequency (str): Data frequency, e.g., "1d" for daily data.
        - time_slice (str): Specific time range for intraday data.
        - expect_df (bool): If True, return the data as a DataFrame.
        - adjust_type (str): Adjustment type for price data (e.g., "none").

        Returns:
        - pd.DataFrame: A DataFrame containing price data with columns:
          ['order_book_id', 'datetime', 'volume', 'close', 'high',
           'trading_date', 'total_turnover', 'open', 'low', 'open_interest'].
        - If no data is available, an empty DataFrame is returned.
        """
        price_data = rqdatac.get_price(
            order_book_ids, start_date, end_date, frequency, adjust_type=adjust_type, time_slice=time_slice, expect_df=expect_df
        )
        if price_data is None:
            return pd.DataFrame()
        return price_data.reset_index()

    def get_vwap(self, tickers: str, start_date: str, end_date: str, frequency="1d"):
        """
        Retrieve VWAP data for specific contracts.

        Parameters:
        - tickers (list or str): List of order book IDs or a single ID.
        - start_date (str): Start date for price data (YYYY-MM-DD).
        - end_date (str): End date for price data (YYYY-MM-DD).
        - frequency (str): Data frequency, e.g., "1d" for daily data.

        Returns:
        - pd.DataFrame: A DataFrame containing price data with columns:
          ['order_book_id', 'datetime', 'volume', 'close', 'high',
           'trading_date', 'total_turnover', 'open', 'low', 'open_interest'].
        - If no data is available, an empty DataFrame is returned.
        """
        price_data = rqdatac.get_vwap(tickers, start_date, end_date, frequency)
        if price_data is None:
            return pd.DataFrame()
        return price_data.reset_index()
