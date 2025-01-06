import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd
import pytz

from common.config import Config, ModelPlatform
from common.database import MsSqlDB, MySqlDB, OracleDB
from common.log import Log

Log = Log()


class TradeDateHelper:
    """
    DateTool provides utility methods for calculating trading dates
    and determining if a given date is a trading day based on exchange calendars.
    """

    def __init__(self):
        self.wind_db = WindDBClient()

    @lru_cache(maxsize=128)
    def _fetch_trade_date(self, market_code: str) -> pd.DataFrame:
        """
        Fetch calendar data for a given calendar type and cache it.
        :param market_code: The type of calendar (e.g., "CFFEX", "DCE").
        :return: DataFrame of the calendar.
        """
        if market_code == "NIB":
            return self.wind_db.get_wind_bond_calendar("NIB")
        elif market_code == "CFFEX" or market_code == "CFE":
            return self.wind_db.get_wind_future_calendar("CFFEX")
        elif market_code == "DCE":
            return self.wind_db.get_wind_future_calendar("DCE")
        else:
            raise ValueError(f"Unknown calendar type: {market_code}")

    def get_trade_date(self, market_code: str) -> pd.DataFrame:
        """
        Retrieve calendar data, using instance-level caching to avoid redundant API calls.
        :param market_code: The type of calendar (e.g., "CFFEX", "DCE").
        :return: DataFrame of the calendar.
        """
        return self._fetch_trade_date(market_code)

    def get_offset_trade_date(self, offset: int, date: str, market_code: str) -> str:
        """
        Calculate a date with an offset based on a given calendar type.
        :param offset: The offset to apply (positive or negative).
        :param date: The reference date as a string in 'YYYYMMDD' format.
        :param market_code: The type of calendar, e.g., "CFFEX", "DCE".
        :return: The calculated date as a string in 'YYYYMMDD' format, or None if out of bounds.
        """
        calendar_df = self.get_trade_date(market_code)
        date = int(date)
        calendar_array = np.array(calendar_df.iloc[:, 0], dtype=int)

        index = np.searchsorted(calendar_array, date, side="left")
        target_index = index + offset if index < len(calendar_array) and calendar_array[index] == date else index - 1 + offset

        if 0 <= target_index < len(calendar_array):
            return str(calendar_array[target_index])
        Log.error("Warning: Calendar range cannot cover the offset date.")
        return None

    def get_last_trade_date(self, date: str, market_code: str) -> str:
        """
        Get the previous trading day for a given exchange.
        :param date: The reference date as a string in 'YYYYMMDD' format.
        :param market_code: The type of calendar (e.g., "CFFEX"/"CFE", "NIB").
        :return: The previous trading day as a string in 'YYYYMMDD' format.
        """
        return self.get_offset_trade_date(-1, date, market_code)

    def is_trade_date(self, date: str, calendar_type: str) -> bool:
        """
        Check if a date is a trading day in a given calendar.
        :param date: The date to check as a string in 'YYYYMMDD' format.
        :param calendar_type: The type of calendar, e.g., "CFFEX", "DCE".
        :return: True if the date is a trading day, False otherwise.
        """
        calendar_df = self.get_trade_date(calendar_type)
        date = int(date)
        calendar_array = np.array(calendar_df.iloc[:, 0], dtype=int)
        return date in calendar_array


class WindDBClient:
    def __init__(self):
        self.wind_db = MsSqlDB(Config.DATABASES["wind"])

    def get_wind_edb(self, indicator_code: str, start_date: str, end_date: str):
        """获取wind EDB指标. 入参:指标代码,开始日期,结束日期,返回指标值和日期"""
        query = """
            SELECT TDATE, INDICATOR_NUM
            FROM gtjaedb
            WHERE F2_4112 = %s AND TDATE BETWEEN %s AND %s
            ORDER BY TDATE
        """
        return self.wind_db.fetch_all(query, (indicator_code, start_date, end_date))

    def get_wind_bond_curve_cnbd(
        self,
        curve_code: int,
        curve_type: int,
        curve_term: float,
        start_date: str,
        end_date: str,
    ):
        """
        获取wind中债登债券收益率曲线. 入参:曲线代码(数值),曲线类型(1:即期,2:到期）,标准期限(年）
        Args:
            curve_code (str): The code for the bond curve.
            curve_type (int): The type of the curve (1 for spot, 2 for maturity).
            curve_term (int): The standard term in years.
            start_date (str): The start date in YYYYMMDD format.
            end_date (str): The end date in YYYYMMDD format.

        Returns:
            list: A list of tuples containing the trade date and yield indicator numbers.
        """

        query = """
            SELECT TRADE_DT AS TDATE, B_ANAL_YIELD AS INDICATOR_NUM
            FROM CBondCurveCNBD
            WHERE B_ANAL_CURVENUMBER = %s AND B_ANAL_CURVETYPE = %s 
                AND B_ANAL_CURVETERM = %s AND TRADE_DT BETWEEN %s AND %s
            ORDER BY TRADE_DT
        """
        return self.wind_db.fetch_all(query, (curve_code, curve_type, curve_term, start_date, end_date))

    def get_wind_bond_futures_price(self, code: str, price_type: str, start_date: str, end_date: str):
        """
        获取wind股指期货日价格. 入参: code: 股指期货wind代码,priceType:str "S_DQ_SETTLE","S_DQ_OPEN","S_DQ_HIGH","S_DQ_LOW"
        """
        query = f"""
            SELECT TRADE_DT, {price_type}
            FROM CBondFuturesEODPrices
            WHERE S_INFO_WINDCODE = %s AND TRADE_DT BETWEEN %s AND %s
            ORDER BY TRADE_DT
        """
        return self.wind_db.fetch_all(query, (code, start_date, end_date))

    def get_wind_commodity_futures_price(self, code: str, price_type: str, start_date: str, end_date: str):
        """
        获取wind商品期货日价格. 入参: code: 商品期货wind代码,priceType:str "S_DQ_SETTLE","S_DQ_OPEN","S_DQ_HIGH","S_DQ_LOW"
        """
        query = f"""
            SELECT TRADE_DT, {price_type}
            FROM CCommodityFuturesEODPrices
            WHERE S_INFO_WINDCODE = %s AND TRADE_DT BETWEEN %s AND %s
            ORDER BY TRADE_DT
        """
        return self.wind_db.fetch_all(query, (code, start_date, end_date))

    def get_wind_index_futures_price(self, code: str, price_type: str, start_date: str, end_date: str):
        """
        获取wind股指期货日价格. 入参: code: 股指期货wind代码,priceType:str "S_DQ_SETTLE","S_DQ_OPEN","S_DQ_HIGH","S_DQ_LOW"
        """
        query = f"""
            SELECT TRADE_DT, {price_type}
            FROM CIndexFuturesEODPrices
            WHERE S_INFO_WINDCODE = %s AND TRADE_DT BETWEEN %s AND %s
            ORDER BY TRADE_DT
        """
        return self.wind_db.fetch_all(query, (code, start_date, end_date))

    def get_wind_bond_calendar(self, market_code: str, start_date: str = None, end_date: str = None):
        """
        获取债券交易日历. 入参: market_code: 交易所市场简称 SSE:上交所 SZSE:深交所 NIB:银行间市场 NBC:柜台交易市场. 
        start_date,endDate格式为 YYYYMMDD
        """
        if start_date is None and end_date is None:
            query = """
                SELECT TRADE_DAYS
                FROM CBONDCALENDAR
                WHERE S_INFO_EXCHMARKET = %s
                ORDER BY TRADE_DAYS
            """
            params = (market_code,)
        elif start_date is None:
            query = """
                SELECT TRADE_DAYS
                FROM CBONDCALENDAR
                WHERE S_INFO_EXCHMARKET = %s AND TRADE_DAYS <= %s
                ORDER BY TRADE_DAYS
            """
            params = (market_code, end_date)
        elif end_date is None:
            query = """
                SELECT TRADE_DAYS
                FROM CBONDCALENDAR
                WHERE S_INFO_EXCHMARKET = %s AND TRADE_DAYS >= %s
                ORDER BY TRADE_DAYS
            """
            params = (market_code, start_date)
        else:
            query = """
                SELECT TRADE_DAYS
                FROM CBONDCALENDAR
                WHERE S_INFO_EXCHMARKET = %s AND TRADE_DAYS BETWEEN %s AND %s
                ORDER BY TRADE_DAYS
            """
            params = (market_code, start_date, end_date)

        return self.wind_db.fetch_all(query, params)

    def get_wind_future_calendar(self, market_code: str, start_date: str = None, end_date: str = None):
        """
        获取期货交易日历. 入参: market_code: 交易所市场简称 SSE:上交所 SZSE:深交所 NIB:银行间市场 NBC:柜台交易市场. 
        start_date,endDate格式为 YYYYMMDD
        """
        if start_date is None and end_date is None:
            query = """
                SELECT TRADE_DAYS
                FROM CFuturesCalendar
                WHERE S_INFO_EXCHMARKET = %s
                ORDER BY TRADE_DAYS
            """
            params = (market_code,)
        elif start_date is None:
            query = """
                SELECT TRADE_DAYS
                FROM CFuturesCalendar
                WHERE S_INFO_EXCHMARKET = %s AND TRADE_DAYS <= %s
                ORDER BY TRADE_DAYS
            """
            params = (market_code, end_date)
        elif end_date is None:
            query = """
                SELECT TRADE_DAYS
                FROM CFuturesCalendar
                WHERE S_INFO_EXCHMARKET = %s AND TRADE_DAYS >= %s
                ORDER BY TRADE_DAYS
            """
            params = (market_code, start_date)
        else:
            query = """
                SELECT TRADE_DAYS
                FROM CFuturesCalendar
                WHERE S_INFO_EXCHMARKET = %s AND TRADE_DAYS BETWEEN %s AND %s
                ORDER BY TRADE_DAYS
            """
            params = (market_code, start_date, end_date)

        return self.wind_db.fetch_all(query, params)

    def get_wind_trade_contract(self, future_code: str, date: str) -> pd.DataFrame:
        """
        获取对应日期可交易合约的代码
        Args:
            future_code (str): 合约类型,10年期国债期货为'T'.
            date (str): The reference date in 'YYYYMMDD' format.

        Returns:
            list: A list of tradable contract codes that match the criteria.
        """

        query = f"""
            SELECT S_INFO_WINDCODE
            FROM CFuturesDescription
            WHERE FS_INFO_SCCODE = %s AND S_INFO_LISTDATE <= %s and S_INFO_DELISTDATE>= %s
            ORDER BY TRADE_DT
        """
        return self.wind_db.fetch_all(query, (future_code, date, date))

    def get_wind_bond_repo_and_ibleod_price(self, price_type: str, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取wind债券同业拆借日行情
        Args:
            price_type (str): The type of price to retrieve (e.g., "S_DQ_SETTLE","S_DQ_OPEN",'S_DQ_HIGH','S_DQ_LOW','B_DQ_WAVERAGERATE').
            code (str): 拆借品种代码
            start_date (str): The start date in 'YYYYMMDD' format.
            end_date (str): The end date in 'YYYYMMDD' format.

        Returns:
            list: A list of tuples containing the trade date and the specified price type.
        """

        query = f"""
            SELECT TRADE_DT, {price_type}
            FROM CFuturesDescription
            WHERE S_INFO_WINDCODE = %s AND TRADE_DT >= %s and TRADE_DT<= %s
            ORDER BY TRADE_DT
        """.format(
            price_type=price_type
        )
        return self.wind_db.fetch_all(query, (code, start_date, end_date))

    def get_wind_c_bond_repo(
        self,
        term: str,
        repo_type: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取wind: CBondRepo
        Args:
            start_date (str): The start date in 'YYYYMMDD' format.
            end_date (str): The end date in 'YYYYMMDD' format.

        Returns:
            list: A list of tuples containing the trade date and the specified bond repo rate.
        """

        query = f"""
            SELECT TRADE_DT as TDATE,B_TENDER_INTERESTRATE as INDICATOR_NUM
            FROM CBondRepo
            WHERE B_INFO_TERM = %s AND B_INFO_REPO_TYPE = %s AND TRADE_DT >= %s AND TRADE_DT <= %s
            ORDER BY TRADE_DT
        """
        return self.wind_db.fetch_all(query, (term, repo_type, start_date, end_date))


class FiccDBClient:
    """FiccDBClient for accessing financial and economic indicator data."""

    def __init__(self):
        self.eficc_db = MySqlDB(Config.DATABASES["eficc"])

    def get_gl_edb(self, indic_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取钢联 EDB指标
        Args:
            indic_code (str): Indicator code.
            start_date (str): Start date in YYYYMMDD format.
            end_date (str): End date in YYYYMMDD format.

        Returns:
            DataFrame: DataFrame containing TDATE and INDICATOR_NUM columns.
        """
        query_sql = """
            SELECT indic_dt AS TDATE, value AS INDICATOR_NUM
            FROM vw_tzyj_indu_indc_value
            WHERE indic_code = %s AND indic_dt BETWEEN %s AND %s
            ORDER BY indic_dt
        """
        try:
            result = self.eficc_db.fetch_all(query_sql, params=(indic_code, start_date, end_date))
            Log.info(f"Retrieved GL EDB data for {indic_code} from {start_date} to {end_date}.")
            return result
        except Exception as e:
            Log.error(f"Error fetching GL EDB data for {indic_code}: {e}")
            raise

    def get_ths_edb(self, indicator_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取同花顺 EDB指标

        Args:
            indicator_id (str): Indicator ID.
            start_date (str): Start date in YYYYMMDD format.
            end_date (str): End date in YYYYMMDD format.

        Returns:
            DataFrame: DataFrame containing TDATE and INDICATOR_NUM columns.
        """
        query_sql = """
            SELECT tm AS TDATE, numerical_value AS INDICATOR_NUM
            FROM industry_eco_index_data
            WHERE indicator_id = %s AND tm BETWEEN %s AND %s
            ORDER BY tm
        """
        try:
            result = self.eficc_db.fetch_all(query_sql, params=(indicator_id, start_date, end_date))
            Log.info(f"Retrieved THS EDB data for {indicator_id} from {start_date} to {end_date}.")
            return result
        except Exception as e:
            Log.error(f"Error fetching THS EDB data for {indicator_id}: {e}")
            raise

    def get_zzcf_ind(self, cf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取中债财富指数指标数据

        Args:
            cf_code (str): Wealth index code.
            start_date (str): Start date in YYYYMMDD format.
            end_date (str): End date in YYYYMMDD format.

        Returns:
            DataFrame: DataFrame containing TDATE and INDICATOR_NUM columns.
        """
        query_sql = """
            SELECT TDATE, CF_VALUE AS INDICATOR_NUM
            FROM wb_index_indicator_zz
            WHERE CF_CODE = %s AND TDATE BETWEEN %s AND %s
            ORDER BY TDATE
        """
        try:
            result = self.eficc_db.fetch_all(query_sql, params=(cf_code, start_date, end_date))
            Log.info(f"Retrieved ZZCF data for {cf_code} from {start_date} to {end_date}.")
            return result
        except Exception as e:
            Log.error(f"Error fetching ZZCF data for {cf_code}: {e}")
            raise


class GsDBClient:
    """GsDBClient for accessing financial and economic indicator data."""

    def __init__(self):
        self.gs_db = OracleDB(Config.DATABASES["gs"])
        self.test_db = OracleDB(Config.DATABASES["test"])

    def get_active_contract_info(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        # Construct SQL query using parameterized inputs
        query_conditions = []
        if start_date:
            query_conditions.append("tdate >= :start_date")
        if end_date:
            query_conditions.append("tdate <= :end_date")

        query_sql = f"""
            SELECT TDATE, MAPPING_CODE AS SYMBOL
            FROM WB_FuturesContractMapping_GS
            WHERE code = :code AND type = '主力合约'
            {('AND ' + ' AND '.join(query_conditions)) if query_conditions else ''}
            ORDER BY tdate
        """

        # Execute query
        params = {"code": code, "start_date": start_date, "end_date": end_date}
        if Config.ModelPlatform == ModelPlatform.LOCAL:
            res = self.test_db.fetch_all(query_sql, params)
        elif Config.ModelPlatform == ModelPlatform.PYTHON_PLATFORM:
            res = self.gs_db.fetch_all(query_sql, params)

        # Process results
        start_date_df = res.groupby(["SYMBOL"]).min()
        end_date_df = res.groupby(["SYMBOL"]).max()
        start_date_df.rename(columns={"TDATE": "START_DATE"}, inplace=True)
        end_date_df.rename(columns={"TDATE": "END_DATE"}, inplace=True)
        df = pd.concat([start_date_df, end_date_df], axis=1, sort=False)
        df.reset_index(inplace=True)

        return df

    def get_dominant_symbol_settle_price_return(
        self,
        symbol: str,
        market_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Calculate settlement price return series for dominant futures contracts.
        :param symbol: DataFrame with 'SYMBOL', 'START_DATE', 'END_DATE'.
        :param market_code: Market identifier ('CFE' or 'DCE').
        :return: Concatenated return series as a DataFrame.
        """
        contract_info = self.get_active_contract_info(symbol, start_date, end_date)

        symbol_list = []
        trade_date_client = TradeDateHelper()
        wind_db = WindDBClient()

        for _, row in contract_info.iterrows():
            symbol_code = row["SYMBOL"]
            pre_date = trade_date_client.get_last_trade_date(row["START_DATE"], market_code)

            # Fetch price data based on market_code
            try:
                if market_code == "CFE":
                    symbol_price_df = wind_db.get_wind_bond_futures_price(symbol_code, "S_DQ_SETTLE", pre_date, row["END_DATE"])
                elif market_code == "DCE":
                    symbol_price_df = wind_db.get_wind_commodity_futures_price(symbol_code, "S_DQ_SETTLE", pre_date, row["END_DATE"])
                else:
                    raise ValueError(f"Unsupported market code: {market_code}")

                # Process price data
                symbol_price_df["S_DQ_SETTLE"] = symbol_price_df["S_DQ_SETTLE"].astype("float64")
                symbol_price_df["TRADE_DT"] = pd.to_datetime(symbol_price_df["TRADE_DT"], format="%Y%m%d")
                symbol_price_series = symbol_price_df.set_index("TRADE_DT")["S_DQ_SETTLE"].rename(symbol_code)
                symbol_list.append(symbol_price_series.pct_change().dropna())

            except Exception as e:
                Log.error(f"Error processing symbol {symbol_code}: {e}")
                continue

        if not symbol_list:
            Log.error("No valid price series found for the given symbol_info.")
            return pd.DataFrame()

        return pd.concat(symbol_list, axis=0)

    def get_dominant_symbol_ohlc(
        self,
        symbol: str,
        market_code: str,
        start_date: str,
        end_date: str,
    ):
        contract_info_df = self.get_active_contract_info(symbol, start_date, end_date)

        symbols_ohlc_price_df = pd.DataFrame()
        wind_db = WindDBClient()

        ohlc_list = ["S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE"]

        for index, row in contract_info_df.iterrows():
            if market_code == "CFE":
                symbol_ohlc_price_df = wind_db.get_wind_bond_futures_price(
                    row["SYMBOL"],
                    "S_DQ_OPEN,S_DQ_HIGH,S_DQ_LOW,S_DQ_CLOSE",
                    row["START_DATE"],
                    row["END_DATE"],
                )
            elif market_code == "DCE":
                symbol_ohlc_price_df = wind_db.get_wind_commodity_futures_price(
                    row["SYMBOL"],
                    "S_DQ_OPEN,S_DQ_HIGH,S_DQ_LOW,S_DQ_CLOSE",
                    row["START_DATE"],
                    row["END_DATE"],
                )

            symbol_ohlc_price_df[ohlc_list] = symbol_ohlc_price_df[ohlc_list].astype("float64")
            symbols_ohlc_price_df = pd.concat([symbols_ohlc_price_df, symbol_ohlc_price_df], axis=0)

        symbols_ohlc_price_df["TRADE_DT"] = symbols_ohlc_price_df.apply(lambda x: dt.datetime.strptime(x["TRADE_DT"], "%Y%m%d"), axis=1)
        symbols_ohlc_price_df.set_index("TRADE_DT", inplace=True)
        symbols_ohlc_price_df.index.name = "dt_ix"
        symbols_ohlc_price_df.rename(
            columns={
                "S_DQ_OPEN": "OPEN",
                "S_DQ_HIGH": "HIGH",
                "S_DQ_LOW": "LOW",
                "S_DQ_CLOSE": "CLOSE",
            },
            inplace=True,
        )
        return symbols_ohlc_price_df

    def get_index_dk_weight(self, calculate_date, index_code):
        query_sql = f"""
            SELECT *
            FROM WB_INDEX_COMPONENT_GS
            WHERE tdate = %s AND inner_code = %s
        """

        if Config.ModelPlatform == ModelPlatform.LOCAL:
            res = self.test_db.fetch_all(query_sql, (calculate_date, index_code))
        elif Config.ModelPlatform == ModelPlatform.PYTHON_PLATFORM:
            res = self.gs_db.fetch_all(query_sql, (calculate_date, index_code))

        return res

    def get_symbol_weight(self, calculate_date, index_code, symbol):
        query_sql = f"""
            SELECT abs(c_weight)
            FROM wb_index_component_gs
            WHERE tdate = %s AND inner_code = %s AND c_code = %s
        """

        if Config.ModelPlatform == ModelPlatform.LOCAL:
            res = self.test_db.fetch_all(query_sql, (calculate_date, index_code, symbol))
        elif Config.ModelPlatform == ModelPlatform.PYTHON_PLATFORM:
            res = self.gs_db.fetch_all(query_sql, (calculate_date, index_code, symbol))

        return res

    def get_mapping_contract(self, code, contract_type, calculate_date):
        query_sql = f"""
            SELECT MAPPING_CODE
            FROM WB_FuturesContractMapping_GS
            WHERE code = %s AND type = %s AND tdate = %s
        """

        if Config.ModelPlatform == ModelPlatform.LOCAL:
            res = self.test_db.fetch_all(query_sql, (code, contract_type, calculate_date))
        elif Config.ModelPlatform == ModelPlatform.PYTHON_PLATFORM:
            res = self.gs_db.fetch_all(query_sql, (code, contract_type, calculate_date))

        return res

    def update_dk_eight(self, index_code: str, calculate_date: str, dk_weight_dic):
        Log.info("指数成分多空权重开始存入数据库")

        update_tm = dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")

        # Iterate through each item in dk_weight_dic to insert or update
        for index, value in dk_weight_dic.items():
            Log.info(f"计算日:{calculate_date}, 成分合约:{index}, 多空权重:{value}")

            data = {
                "tdate": calculate_date,
                "inner_code": index_code,
                "c_code": index,
                "c_market_code": "CFFEX",
                "c_asset_type": "FUT_BD",
                "c_weight": value,
                "update_tm": update_tm,
            }

            condition = "WHERE inner_code = :inner_code AND tdate = :tdate"

            try:
                if Config.ModelPlatform == ModelPlatform.LOCAL:
                    self.test_db.upsert_data("wb_index_component_gs", data, condition)
                elif Config.ModelPlatform == ModelPlatform.PYTHON_PLATFORM:
                    self.gs_db.upsert_data("wb_index_component_gs", data, condition)

                Log.info(f"Data for {index} successfully upserted.")
            except Exception as e:
                Log.error(f"Error upserting data for {index}: {e}")

        Log.info("指数成分多空权重存入数据库结束")

    def update_index_value(self, index_code: str, calculate_date: str, index_value: float):
        Log.info("指数点位开始存入数据库")

        # Log the details of the calculation
        Log.info(f"计算日:{calculate_date}, 指数点位: {index_value}")

        # Get the current time for the update
        update_tm = dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")

        # Define the SQL query for merge (upsert) operation using placeholders
        sql = """
        MERGE INTO gs.wb_index_result_gs t1
        USING (
            SELECT COUNT(*) c 
            FROM gs.wb_index_result_gs t 
            WHERE t.tdate = :calculate_date AND t.inner_code = :index_code
        ) t2
        ON (t2.c > 0)
        WHEN MATCHED THEN
            UPDATE
            SET t1.value = :index_value,
                t1.update_tm = TO_DATE(:update_tm, 'yyyy-mm-dd hh24:mi:ss')
            WHERE t1.tdate = :calculate_date AND t1.inner_code = :index_code
        WHEN NOT MATCHED THEN
            INSERT (t1.tdate, t1.inner_code, t1.value, t1.update_tm)
            VALUES (:calculate_date, :index_code, :index_value, TO_DATE(:update_tm, 'yyyy-mm-dd hh24:mi:ss'))
        """

        params = {"calculate_date": calculate_date, "index_code": index_code, "index_value": index_value, "update_tm": update_tm}

        try:
            if Config.ModelPlatform == ModelPlatform.LOCAL:
                self.test_db.execute_query(sql, params)
            elif Config.ModelPlatform == ModelPlatform.PYTHON_PLATFORM:
                self.gs_db.execute_query(sql, params)

            Log.info("指数点位存入数据库结束")
        except Exception as e:
            Log.error(f"Error updating index value: {e}")
            raise
