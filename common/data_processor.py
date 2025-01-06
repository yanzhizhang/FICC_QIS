# data_processor.py

import datetime as dt
import os

import pandas
import pandas as pd

from common.config import Config, ModelPlatform
from common.database import OracleDB
from common.database_client import (
    FiccDBClient,
    GsDBClient,
    TradeDateHelper,
    WindDBClient,
)
from common.log import Log

Log = Log()


class DataProcessor:
    def __init__(self, calculate_date):
        self.wind_db_client = WindDBClient()
        self.ficc_db_client = FiccDBClient()
        self.trade_date_client = TradeDateHelper()
        self.gs_db_client = GsDBClient()

        self.today_date = calculate_date
        self.start_date = "20000101"

        self.today_dt = dt.datetime.strptime(self.today_date, "%Y%m%d")

        # 因子: 9 month yield
        self.yield9m_series = self._get_yield9m()
        # 因子: zz_caifu_return_series
        self.zz_caifu_return_series = None
        self._get_zz_caifu()
        # 因子: t_ohlc_df, t_return_series
        self.t_return_series = None
        self.t_ohlc_df = None
        self._get_t()

        # 模型 2
        # 因子1 luowengang_kucun
        self.luowengang_series = None
        self._get_luowengang_kucun()
        # 因子2 tangshangaolu
        self.tangshangaolu_series = None
        self._get_tangshangaolu()
        # 因子3 jiaotan
        self.jiaotan_df = None
        self.jiaotan_return_series = None
        self._get_jiaotan()
        # 因子 4 m1 cash
        self.m1_series = None
        self._get_m1()
        # 因子5 mianji_30
        self.mianji30_series = None
        self._get_mianji_30()

        # 模型 3
        # 因子6 nrepo
        self.reverse_repo_7_s = self._get_reverse_repo("7")
        # 因子7 yield10y
        self.yield10y_s = self._get_yield10y()

    def _load_csv_data(self, file_name):
        """Helper method to load CSV data from the local file system."""
        # __file__ 在被继承之后依然返回本脚本的目录 `./common`
        file_path = os.path.join(os.path.dirname(__file__), "..", "data", file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    def _convert_time_series(self, df: pd.DataFrame, col_name: str):
        """Helper method to convert dataframe with a date column to time-series."""
        try:
            # Ensure required columns are present
            if "TDATE" not in df.columns or "INDICATOR_NUM" not in df.columns:
                raise ValueError("DataFrame must contain 'TDATE' and 'INDICATOR_NUM' columns.")
            # Convert TDATE to datetime
            df["TDATE"] = pd.to_datetime(df["TDATE"], format="%Y%m%d")
            # Set TDATE as the index
            df.set_index("TDATE", inplace=True)
            # Extract and return the INDICATOR_NUM column as a time-indexed series
            return df["INDICATOR_NUM"].rename(col_name)
        except Exception as e:
            Log.error(f"Error in _convert_time_series: {str(e)}")
            raise

    def _get_m1(self):
        """
        因子: M1
            WindEDB: M0001383
            开始日期: 2003.1.31

            无
        """
        m1_df = self.wind_db_client.get_wind_edb("M0001383", self.start_date, self.today_date)
        self.m1_series = self._convert_time_series(m1_df, "M1")

    def _get_mianji_30(self):
        """
        因子: mianji_30
            WindEDB: S2707380
            开始日期: 2010.1.1

            无
        """
        mianji_30_df = self.wind_db_client.get_wind_edb("S2707380", self.start_date, self.today_date)
        self.mianji30_series = self._convert_time_series(mianji_30_df, "MIANJI30")

    def _get_luowengang_kucun(self):
        """
        因子: luowengang_kucun
            WindEDB:S0181750
            开始日期: 2010.5.21
            2012.1.5 后使用钢联数据库
        """
        # 螺纹钢 2011年12月31日 以前数据
        luowengang_wind_csv = self._load_csv_data("luowengang_wind_his.csv")

        # Additional processing for local or platform configurations
        if Config.ModelPlatform == ModelPlatform.LOCAL:
            luowengang_gl = self._load_csv_data("luowengang_gl.csv")
        else:
            luowengang_gl = self.ficc_db_client.get_gl_edb("ID0300000018", self.start_date, self.today_date)

        # Convert "TDATE" to integers for comparison
        luowengang_gl["TDATE"] = luowengang_gl["TDATE"].astype(int)

        # Filter data after 2012-01-01
        luowengang_gl = luowengang_gl[luowengang_gl["TDATE"] >= 20120101]

        # Concatenate and reindex
        luowengang_df = pd.concat([luowengang_wind_csv, luowengang_gl], axis=0).reset_index(drop=True)

        luowengang_df = luowengang_df[luowengang_df["TDATE"] <= int(self.today_date)]

        self.luowengang_series = self._convert_time_series(luowengang_df, "LUOWENGANG")

    def _get_tangshangaolu(self):
        """
        因子: tangshangaolu
            WindEDB: S5711217
            开始日期: 2013.12.20
            WindEDB: S5707135
            开始日期: 2009.1.9
        """
        s5711217 = self.wind_db_client.get_wind_edb("S5711217", self.start_date, self.today_date)
        s5711217["TDATE"] = s5711217["TDATE"].astype(int)
        s5711217 = s5711217[s5711217["TDATE"] >= 20131220]

        s5707135 = self.wind_db_client.get_wind_edb("S5707135", self.start_date, self.today_date)
        s5707135["TDATE"] = s5707135["TDATE"].astype(int)
        s5707135 = s5707135[(s5707135["TDATE"] >= 20090109) & (s5707135["TDATE"] <= 20131220)]

        # Concatenate both datasets and filter by TDATE <= today_date
        tangshangaolu_df = pd.concat([s5707135, s5711217], axis=0).reset_index(drop=True)

        tangshangaolu_df = tangshangaolu_df[tangshangaolu_df["TDATE"] <= int(self.today_date)]

        self.tangshangaolu_series = self._convert_time_series(tangshangaolu_df, "TANGSHANGAOLU")

    def _get_reverse_repo(self, term: str = "7"):
        """
        因子: nrepo
            WindEDB: M0041371
            开始日期: 2002.1.8

            无
        """
        reverse_repo_rate = self.wind_db_client.get_wind_c_bond_repo(
            term=term,
            repo_type="517002000",
            start_date=self.start_date,
            end_date=self.today_date,
        )
        reverse_repo_rate["INDICATOR_NUM"] = reverse_repo_rate["INDICATOR_NUM"].astype("float64")
        return self._convert_time_series(reverse_repo_rate, "R_REPO")

    def _get_bond_yield(self, term: float | int) -> pandas.Series:
        """
        从 wind 落地数据库获取国债到期收益率

        :param term: The term parameter in the _get_bond_yield method represents the term to maturity of
        a bond, which is the length of time until the bond reaches its maturity date. This term is
        typically expressed in years as a floating-point number or an integer
        :type term: float|int
        """
        bond_yield = self.wind_db_client.get_wind_bond_curve_cnbd(1232, 2, term, self.start_date, self.today_date)
        bond_yield["INDICATOR_NUM"] = bond_yield["INDICATOR_NUM"].astype("float64")
        return self._convert_time_series(bond_yield, "YIELD")

    def _get_yield9m(self):
        """
        因子: yield9m
            WindEDB: M1000157
            开始日期: 2002.1.8

            无
        """
        return self._get_bond_yield(0.75).rename("YIELD9M")

    def _get_yield10y(self):
        """
        因子: yield10y
            WindEDB: M0041371
            开始日期: 2002.1.8

            无
        """
        return self._get_bond_yield(10).rename("YIELD10Y")

    def _get_t(self):
        """
        因子: t_ohlc
            WindEDB: M1000157
            开始日期: 2002.1.8

            无
        """
        # 国债期货主力合约信息
        self.t_return_series = self.gs_db_client.get_dominant_symbol_settle_price_return(
            symbol="T.CFE",
            market_code="CFE",
            start_date=self.start_date,
            end_date=self.today_date,
        ).rename("10YRETURN")

        self.t_ohlc_df = self.gs_db_client.get_dominant_symbol_ohlc(
            symbol="T.CFE",
            market_code="CFE",
            start_date=self.start_date,
            end_date=self.today_date,
        )

    def _get_jiaotan(self):
        """
        因子: jiaotan
            WindWSD:J(焦炭)
                J.DCE

            (1)按照主力合约列表,拼接构造结算价收益率序列;
            (2)前步骤序列+1后计算累计乘积序列;
            (3)使用前步骤序列月度最后值进行月度重采样;
            (4)转换同比
        """
        self.jiaotan_df = self.gs_db_client.get_active_contract_info(code="J.DCE", start_date=self.start_date, end_date=self.today_date)

        if self.jiaotan_df is None or self.jiaotan_df.empty:
            Log.error("Failed to retrieve active contract information for J.DCE")
            return

        # Calculate dominant contract settlement price returns
        self.jiaotan_return_series = self.gs_db_client.get_dominant_symbol_settle_price_return(
            "J.DCE", "DCE", start_date=self.start_date, end_date=self.today_date
        )
        self.jiaotan_return_series.rename("JIAOTAN", inplace=True)

    def _get_zz_caifu(self):
        """
        因子: zz_caifu
        """
        # Load static historical data
        zz_caifu_return_old_df = self._load_csv_data("zz_dataold.csv")

        # Filter by date and adjust old data
        zz_caifu_return_old_df.iloc[:, 0] = zz_caifu_return_old_df.iloc[:, 0].astype(str)
        zz_caifu_return_old_df = zz_caifu_return_old_df[zz_caifu_return_old_df.iloc[:, 0] <= self.today_date]
        zz_caifu_return_old_series = self._adjust_date_series(zz_caifu_return_old_df, "caifu") / 100

        # Load recent data based on platform configuration
        if Config.ModelPlatform == ModelPlatform.LOCAL:
            zz_caifu_df = pd.read_csv("zz_data.csv")
        else:
            zz_caifu_df = self.ficc_db_client.get_zzcf_ind("CBF00811", "20140901", self.today_date)
            zz_caifu_df["INDICATOR_NUM"] = zz_caifu_df["INDICATOR_NUM"].astype("float64")

        # Filter by date and adjust recent data
        zz_caifu_df.iloc[:, 0] = zz_caifu_df.iloc[:, 0].astype(str)
        zz_caifu_df = zz_caifu_df[zz_caifu_df.iloc[:, 0] <= self.today_date]
        zz_caifu_series = self._adjust_date_series(zz_caifu_df, "caifu")

        # Compute percentage change for recent data
        zz_caifu_return_series = zz_caifu_series.pct_change().dropna()

        # Combine old and recent return series
        self.zz_caifu_return_series = pd.concat([zz_caifu_return_old_series, zz_caifu_return_series]).dropna()

    def _adjust_date_series(self, in_data: pd.DataFrame, name: str) -> pd.Series:
        """
        Adjust a DataFrame into a time-indexed Series.
        :param in_data: Input DataFrame with date in the first column and values in the second column.
        :param name: Name for the resulting Series.
        :return: A time-indexed Series.
        """
        in_data.iloc[:, 0] = pd.to_datetime(in_data.iloc[:, 0], format="%Y%m%d")
        return in_data.set_index(in_data.iloc[:, 0])[in_data.columns[1]].rename(name)

    def get_mapping_contract(code: str, type: str, calculate_date: str):
        # 获取计算日期、类型对应的映射合约
        sql = """
            select MAPPING_CODE from gs.WB_FuturesContractMapping_GS where code='{code}' and 
            type='{type}' and tdate='{calculate_date}' 
            """.format(
            code=code, calculate_date=calculate_date, type=type
        )

        if Config.ModelPlatform == 1:
            oracle_db = OracleDB(Config.DATABASES["gs"])
            res = oracle_db.read_data(sql)
        else:
            oracle_db = OracleDB(Config.DATABASES["test"])
            res = oracle_db.read_data(sql)

        if res.empty:
            Log.error(
                "指定日期和类型无对应映射合约",
                "连续主力合约代码=",
                code,
                ",类型=",
                type,
                ",日期=",
                calculate_date,
            )
        contract = res.iloc[0, 0]
        return contract
