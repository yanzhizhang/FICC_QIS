import datetime as dt
import os

import pandas as pd

from common.config import Config, ModelPlatform
from common.database_client import GsDBClient, TradeDateHelper
from common.log import Log
from common.minio import Minio

Log = Log()


class Weight:
    def __init__(self, calculate_date):
        self.calculate_date = calculate_date
        trade_date_client = TradeDateHelper()
        self.calculate_date = dt.datetime.strptime(calculate_date, "%Y%m%d").date()
        self.cffex_prev_day = trade_date_client.get_last_trade_date(self.today_date, "CFFEX")

    def get_weight(self):
        weight_dic = {}

        if self.calculate_date < dt.date(2015, 8, 1):
            weight_dic["T1509.CFE"] = 1
            return weight_dic

        gs_client = GsDBClient()
        pre_dk_weight = gs_client.get_index_dk_weight(self.cffex_prev_day, "Z0000")
        if pre_dk_weight.empty:
            Log.error("前一交易日权重未更新,前一交易日期:", self.cffex_prev_day)
            return None

        his_roll_record_df = self.get_his_roll_record()
        his_roll_start_dt = dt.datetime.strptime(str(his_roll_record_df["STARTDATE"].iloc[-1]), "%Y%m%d").date()
        his_roll_end_dt = dt.datetime.strptime(str(his_roll_record_df["ENDDATE"].iloc[-1]), "%Y%m%d").date()

        if self.calculate_date < his_roll_start_dt:
            Log.error("计算日在最后一次展期日期前,", "最后一次展期开始日=", his_roll_record_df["STARTDATE"].iloc[-1], ",计算日=", self.calculate_date)
            return None
        elif self.calculate_date <= his_roll_end_dt:
            return self._process_within_roll_period(his_roll_record_df, weight_dic)
        else:
            return self._process_outside_roll_period(his_roll_record_df, weight_dic)

    def _process_within_roll_period(self, his_roll_record_df, weight_dic):
        old_symbol = his_roll_record_df["OLDSYMBOL"].iloc[-1]
        new_symbol = his_roll_record_df["NEWSYMBOL"].iloc[-1]

        gs_client = GsDBClient()

        old_symbol_preweight = gs_client.get_symbol_weight(self.cffex_prev_day, "Z0000", old_symbol)
        new_symbol_preweight = gs_client.get_symbol_weight(self.cffex_prev_day, "Z0000", new_symbol)

        if new_symbol_preweight is None:
            weight_dic[old_symbol] = old_symbol_preweight
            weight_dic[new_symbol] = 0
            Log.info("计算日处于展期开始日,为新合约权重赋初值0.", "新合约权重=", new_symbol_preweight, ",旧合约权重=", old_symbol_preweight)
            return weight_dic

        weight_dic[old_symbol] = old_symbol_preweight - 0.2
        weight_dic[new_symbol] = new_symbol_preweight + 0.2
        return weight_dic

    def _process_outside_roll_period(self, his_roll_record_df, weight_dic):
        if self.calculate_date.month in [2, 5, 8, 11] and self.calculate_date.day >= 15:
            now_roll_symbol = his_roll_record_df["NEWSYMBOL"].iloc[-1]

            gs_client = GsDBClient()

            if self.calculate_date.month + 1 == self.get_contract_month(now_roll_symbol):
                old_symbol = now_roll_symbol
                new_symbol = gs_client.get_mapping_contract("T.CFE", "次主力合约", self.cffex_prev_day)
                roll_record_dic = self.add_roll_record(self.calculate_date, old_symbol, new_symbol)
                roll_record_df = his_roll_record_df._append(roll_record_dic, ignore_index=True)
                self.update_roll_record(roll_record_df)
                weight_dic[now_roll_symbol] = 1
                return weight_dic

        new_symbol = his_roll_record_df["NEWSYMBOL"].iloc[-1]
        weight_dic[new_symbol] = 1
        return weight_dic

    def get_his_roll_record(self):
        if Config.ModelPlatform == ModelPlatform.LOCAL:
            path = os.path.join(os.path.dirname(__file__), "..", "data", "roll_record.xlsx")
            return pd.read_excel(path)
        else:
            file_name = "roll_record.xlsx"
            Minio.download_from_remote(file_name, Config.GS_MINIO_PATH)
            return pd.read_excel(file_name)

    def update_roll_record(self, df):
        Log.info("展期记录表有更新")
        if Config.ModelPlatform == ModelPlatform.LOCAL:
            path = os.path.join(os.path.dirname(__file__), "..", "data", "roll_record.xlsx")
            df.to_excel(path, index=False)
        else:
            file_name = "roll_record.xlsx"
            df.to_excel(file_name, index=False)
            Minio.save_to_remote(file_name, Config.GS_MINIO_PATH)

    def add_roll_record(self, calculate_date, old_symbol, new_symbol):
        trade_date_client = TradeDateHelper()
        roll_start_date = trade_date_client.get_offset_trade_date(calculate_date, "CFFEX", 1)
        roll_end_date = trade_date_client.get_offset_trade_date(calculate_date, "CFFEX", 6)
        return {"STARTDATE": roll_start_date, "ENDDATE": roll_end_date, "OLDSYMBOL": old_symbol, "NEWSYMBOL": new_symbol}

    def get_contract_month(self, code):
        index = code.find(".")
        return int(code[index - 2 : index])
