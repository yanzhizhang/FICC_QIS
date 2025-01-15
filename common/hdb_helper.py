import hdb
import numpy as np
from common.config import Config

import pandas as pd
from common.log import Log

Log = Log()


class HdbHelper:
    def __init__(self):
        """
        Initialize the HdbApi client with configuration details.
        """
        self.host = Config.DATABASES["hdb"]["host"]
        self.port = Config.DATABASES["hdb"]["port"]
        self.username = Config.DATABASES["hdb"]["username"]
        self.password = Config.DATABASES["hdb"]["password"]
        self.client = hdb.Client(self.host, self.port, self.username, self.password)

    def get_bar_data(self, symbol_list, file_path):
        """
        Retrieve minute-level candlestick data for specified symbols.

        Args:
        - symbol_list (list): List of symbols to retrieve data for.
        - file_path (str): Path to the HDB file.

        Returns:
        - pd.DataFrame: DataFrame containing K-line data.
        """
        k_bar_type = ["SecurityKdata"]
        file = self.client.open_file(file_path)

        task = file.open_read_task(symbols=symbol_list, types=k_bar_type)

        items = task.read()

        items_data = file.data_types[0].items_data(items["data"])
        k_bar_dict = {
            "HDB_SYMBOL": np.char.decode(items["symbol"]),
            "HDB_DATE": items["trading_day"],
            "HDB_TIME": items_data["time"],
            "HDB_OPEN": items_data["open"],
            "HDB_HIGH": items_data["high"],
            "HDB_LOW": items_data["low"],
            "HDB_CLOSE": items_data["close"],
            "HDB_VOLUMN": items_data["volume"],
            "HDB_TURNOVER": items_data["turnover"],
        }
        return pd.DataFrame.from_dict(k_bar_dict)

    # def get_tick_data(self, symbol_list, file_path):
    #     """
    #     Retrieve tick-level data for specified symbols.

    #     Args:
    #     - symbol_list (list): List of symbols to retrieve data for.
    #     - file_path (str): Path to the HDB file.

    #     Returns:
    #     - pd.DataFrame: DataFrame containing tick data.
    #     """
    #     k_bar_type = ["FuturesTick"]
    #     file = self.client.open_file(file_path)
    #     task = file.open_read_task(symbols=symbol_list, types=k_bar_type)
    #     items = task.read()
    #     items_data = file.data_types[2].items_data(items["data"])
    #     k_bar_dict = {
    #         "symbol": np.char.decode(items["symbol"]),
    #         "trading_day": items["trading_day"],
    #         "time": items_data["time"],
    #         "open": items_data["open"],
    #         "high": items_data["high"],
    #         "low": items_data["low"],
    #         "close": items_data["close"],
    #         "volume": items_data["volume"],
    #         "turnover": items_data["turnover"],
    #     }
    #     return pd.DataFrame.from_dict(k_bar_dict)

    def transform_k_data(self, df, resample_type):
        """
        Transform high-frequency K-line data to low-frequency data.

        Args:
        - df (pd.DataFrame): DataFrame with high-frequency K-line data.
        - resample_type (str): Resampling frequency (e.g., 'H', '30min').

        Returns:
        - pd.DataFrame: Resampled DataFrame.
        """
        df["trading_day"] = df["trading_day"].astype("str")
        df["time"] = df["time"].astype("str")
        df["time"] = df["time"].apply(lambda x: x if len(x) == 4 else "0" + x)
        df["datetime"] = pd.to_datetime(df["trading_day"] + " " + df["time"])
        df.set_index("datetime", inplace=True)
        df.drop(["time"], axis=1, inplace=True)
        res_df = (
            df.groupby("symbol")
            .resample(resample_type, closed="right", label="right")
            .agg({"trading_day": "first", "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "turnover": "sum"})
            .dropna()
        )
        res_df.reset_index(inplace=True)
        return res_df

    def map_wind_to_hdb_codes(self, wind_code_list):
        """
        Convert Wind codes to HDB-compatible codes in batch.

        Args:
        - wind_code_list (list): List of Wind contract codes.

        Returns:
        - pd.DataFrame: DataFrame mapping Wind codes to HDB codes.
        """
        excode_map = {"CZC": "CZCE", "SHF": "SHFE", "DCE": "DCE", "INE": "INE", "GFE": "GFEX"}
        low_case_exchanges = ["SHF", "DCE", "INE", "GFE"]
        high_case_exchanges = ["CZC"]
        hdb_code_list = []

        for wind_code in wind_code_list:
            parts = wind_code.split(".", 1)
            excode_hdb = excode_map.get(parts[1])
            if excode_hdb is None:
                Log.error(f"Mapping not found for exchange Wind code: {parts[1]}")
                continue
            contract_code_hdb = parts[0].lower() if parts[1] in low_case_exchanges else parts[0]
            hdb_code_list.append(f"{excode_hdb}.{contract_code_hdb}")

        data = {"WINDCODE": wind_code_list, "HDBCODE": hdb_code_list}
        return pd.DataFrame(data)
