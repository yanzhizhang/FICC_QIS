import datetime as dt

import pandas as pd
import pytz
import rqdatac


class FuturePriceRetriever:
    def __init__(self, start_date: str = "20100104", end_date: str = None):
        """
        Initializes the FuturePriceRetriever class with start and end dates.

        Parameters:
        - start_date: The start date for data retrieval in YYYYMMDD format (default is "20100104").
        - end_date: The end date for data retrieval in YYYYMMDD format (default is today's date).
        """
        if start_date < "20100104":
            raise ValueError("start_date must be larger than or equal to 20100104")
        self.start_date = start_date
        self.end_date = end_date if end_date else dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d")
        rqdatac.init()

    def get_future_price_data(
        self,
        symbols,
        frequency="1d",
        fields=None,
        adjust_type="pre",
        adjust_method="prev_close_spread",
    ):
        """
        Retrieves the dominant future price data for given symbols from a specified time range.

        Parameters:
        - symbols: A list of future symbols to retrieve data for (e.g., ['RB', 'HC']).

        Returns:
        - A DataFrame containing the price data of the given symbols.

        Example:
        rqdatac.futures.get_dominant_price(
            underlying_symbols='IF',
            start_date=20210901,
            end_date=20210902,
            frequency='1d',
            fields=None,
            adjust_type='pre',
            adjust_method='prev_close_spread'
        )
        Output:
                                        dominant_id     open    close    high     low  total_turnover    volume  prev_close  settlement  prev_settlement  open_interest  limit_up  limit_down  day_session_open
        underlying_symbol   date
        IF                  2021-09-01  IF2109          4577.4  4666.6  4708.8  4547.4               0  130017.0      4588.6      4665.2           4577.0       143730.0    5053.6      4100.4            4767.2
        IF                  2021-09-02  IF2109          4665.2  4664.4  4689.2  4640.4               0   73853.0      4666.6      4664.2           4665.2       128436.0    5150.6      4179.8            4855.0

        rqdatac.futures.get_dominant_price(
            underlying_symbols="IF",
            start_date=20210901,
            end_date=20210901,
            frequency="1m",
            fields=None,
            adjust_type="none",
            adjust_method="prev_close_spread",
        )
        Output:
        underlying_symbol   datetime            trading_date dominant_id    open    close    high     low  total_turnover    volume  open_interest
        IF                  2021-09-01 09:31:00 2021-09-01   IF2109        4767.2  4779.0  4781.8  4767.0    2.990109e+09    2087.0       140180.0
                            2021-09-01 09:32:00 2021-09-01   IF2109        4779.6  4773.2  4780.6  4772.0    1.496143e+09    1044.0       139408.0
                            2021-09-01 09:33:00 2021-09-01   IF2109        4773.2  4763.4  4773.2  4763.0    1.379020e+09     964.0       138709.0
                            2021-09-01 09:34:00 2021-09-01   IF2109        4763.4  4751.4  4763.4  4750.8    1.768014e+09    1239.0       137894.0
                            2021-09-01 09:35:00 2021-09-01   IF2109        4750.8  4755.2  4755.6  4748.0    1.605260e+09    1126.0       137099.0
                            ...                                            ...     ...     ...     ...             ...       ...            ...
                            2021-09-01 14:56:00 2021-09-01   IF2109        4857.2  4852.6  4857.2  4852.2    6.394091e+08     439.0       142914.0
                            2021-09-01 14:57:00 2021-09-01   IF2109        4852.2  4854.8  4858.0  4851.8    6.219307e+08     427.0       143000.0
                            2021-09-01 14:58:00 2021-09-01   IF2109        4854.4  4855.2  4855.4  4851.0    8.574656e+08     589.0       143113.0
                            2021-09-01 14:59:00 2021-09-01   IF2109        4855.0  4856.8  4858.2  4853.8    6.483443e+08     445.0       143410.0
                            2021-09-01 15:00:00 2021-09-01   IF2109        4857.4  4856.4  4861.0  4855.8    8.907135e+08     611.0       143730.0
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
        return price_df

    def get_spread_data(
        self,
        symbols,
        frequency="1d",
        fields=None,
        adjust_type="pre",
        adjust_method="prev_close_spread",
    ):
        """
        Calculates the spread between two symbols (e.g., RB and HC) and returns the spread data.

        Parameters:
        - symbols: A list of two future symbols (e.g., ['RB', 'HC']).

        Returns:
        - A DataFrame containing the spread between the given symbols.
        """
        price_df = self.get_future_price_data(
            symbols,
            frequency=frequency,
            fields=fields,
            adjust_type=adjust_type,
            adjust_method=adjust_method,
        )
        if len(symbols) == 2:
            spread_df = pd.DataFrame(
                {
                    f"{symbols[1]}_prices": price_df[symbols[1]],
                    f"{symbols[0]}_prices": price_df[symbols[0]],
                    f"{symbols[0]}_{symbols[1]}_spread": price_df[symbols[0]] - price_df[symbols[1]],
                }
            )
            return spread_df
        else:
            raise ValueError("The 'symbols' list must contain exactly two symbols to calculate spread.")
