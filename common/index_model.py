# index_model.py

import datetime as dt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

from common.log import Log

Log = Log()


class BaseIndexModel:
    def __init__(self):
        pass

    def get_train_end_date(self, target_dt):
        target_month = target_dt.month
        month_dic = {
            1: 9,
            2: 12,
            3: 12,
            4: 12,
            5: 3,
            6: 3,
            7: 3,
            8: 6,
            9: 6,
            10: 6,
            11: 9,
            12: 9,
        }
        train_month = month_dic[target_month]
        if target_month < train_month:
            return dt.date(target_dt.year - 1, train_month, 2)  # 前面的都是月频数据,1号,截止训练日期定为2号
        else:
            return dt.date(target_dt.year, train_month, 2)

    def get_target_dt(self):
        cffex_prev_dt = dt.datetime.strptime(self.cffex_prev_day, "%Y%m%d")
        first_day_of_month = self.today_dt.replace(day=1)

        # 判断CFFEX的前一个交易日是否在同一个月
        if cffex_prev_dt.month == self.today_dt.month:
            return first_day_of_month + relativedelta(months=1)
        else:
            return first_day_of_month

    def timing_bt_regress(self, px_return_s: pd.Series, factor_s: pd.Series, month_window: int = 1):
        """
        Perform a rolling regression to generate timing signals based on factor performance.

        :param px_return_s: Series of price returns with datetime index.
        :param factor_s: Series of factor values with datetime index.
        :return: Dictionary with signals and regression model.
        """
        # Prepare the regression DataFrame
        regress_df = pd.concat([factor_s, px_return_s], axis=1).dropna()
        regress_df.columns = ["factor", "returns"]

        # Assign month groups
        month_groups = regress_df.index.to_period("M")

        # Initialize outputs
        predict_y_s = pd.Series(index=regress_df.index, dtype=float)
        last_month_i = 0
        reg_res = None

        # Iterate over unique months
        for month in month_groups.unique()[month_window:]:

            # training_months = month_groups.unique()[month_groups.unique().get_loc(month) - month_window : month_groups.unique().get_loc(month)]
            # training_indices = regress_df[month_groups.isin(training_months)].index
            # Y = regress_df.loc[training_indices, ["returns"]]
            # x = sm.add_constant(regress_df.loc[training_indices, ["factor"]], has_constant="add")

            # Perform regression on data up to the previous month
            current_month_i = regress_df[month_groups == month].index
            Y = regress_df.loc[:current_month_i[0], ["returns"]]
            x = sm.add_constant(regress_df.loc[:current_month_i[0], ["factor"]], has_constant="add")

            reg_model = sm.OLS(
                Y,
                x,
                hasconst=True,
            )
            reg_res = reg_model.fit()

            # Predict for the current month's data
            predictions = reg_res.predict(sm.add_constant(regress_df.loc[current_month_i, ["factor"]], has_constant="add"))

            predict_y_s.loc[current_month_i] = predictions

            last_month_i = current_month_i[-1]

        # Generate signals
        signal_s = predict_y_s.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        return {"signal_s": signal_s, "reg_model": reg_res}

    def process_factor(self, series, resample_method="last", lag=0, pct_change_periods=0, rolling_window=None):
        """
        Generalized method to process factors with resampling, lagging, and transformations.

        Args:
            series (pd.Series): Input time series.
            resample_method (str): Method to aggregate monthly data ('last', 'mean', 'sum', 'first').
            lag (int): Number of periods to lag the series.
            pct_change_periods (int): Number of periods for percentage change calculation.
            rolling_window (int): Window size for smoothing, if needed.

        Returns:
            pd.Series: Transformed and processed time series.
        """
        # Resampling and forward-filling
        if resample_method == "last":
            series = series.resample("M").last().ffill()
        elif resample_method == "mean":
            series = series.resample("M").mean().ffill()
        elif resample_method == "sum":
            series = series.resample("M").sum().ffill()
        elif resample_method == "first":
            series = series.resample("M").first().ffill()
        else:
            raise ValueError("resample_method should be one of ('last', 'mean', 'sum', 'first').")

        # Ensure index is set to the first day of each month
        series.index = series.index.to_period("M").to_timestamp("M") + pd.Timedelta(days=1) - pd.Timedelta(days=1)
        series.index = series.index - pd.offsets.MonthBegin(1)

        # Percentage change calculation
        if pct_change_periods > 0:
            series = series.astype(float).ffill()
            series = series.pct_change(pct_change_periods)

        # Apply rolling mean if specified
        if rolling_window:
            series = series.rolling(rolling_window).mean()

        return series
    
    def calc_dominant_param(self, return_df: pd.DataFrame):
        dominant_weight_dict = {"dt_ix": [], "symbol": [], "weight": []}
        dominant_val_list = []
        index_list = []
        last_row_i = -1

        for col_i in range(1, return_df.shape[1]):
            weight = 0.2
            for row_i in range(last_row_i + 1, return_df.shape[0]):
                if weight > 1:
                    break
                if np.isnan(return_df.iloc[row_i, col_i]):
                    dominant_val_list.append(return_df.iloc[row_i, col_i - 1])
                    dominant_weight_dict["dt_ix"].append(return_df.index[row_i])
                    dominant_weight_dict["symbol"].append(return_df.columns[col_i - 1])
                    dominant_weight_dict["weight"].append(1)
                else:
                    dominant_val_list.append((1 - weight) * return_df.iloc[row_i, col_i - 1] + weight * return_df.iloc[row_i, col_i])
                    dominant_weight_dict["dt_ix"].append(return_df.index[row_i])
                    dominant_weight_dict["symbol"].append(return_df.columns[col_i - 1])
                    dominant_weight_dict["weight"].append(1 - weight)
                    dominant_weight_dict["dt_ix"].append(return_df.index[row_i])
                    dominant_weight_dict["symbol"].append(return_df.columns[col_i])
                    dominant_weight_dict["weight"].append(weight)
                    weight += 0.2
                last_row_i = row_i
                index_list.append(return_df.index[row_i])

        return pd.Series(dominant_val_list, index=index_list), pd.DataFrame(dominant_weight_dict).set_index(["dt_ix", "symbol"]).unstack().droplevel(
            level=0, axis=1
        )

    def calc_dominant_param2(self, return_df: pd.DataFrame):
        dominant_weight_dict = {"dt_ix": [], "symbol": [], "weight": []}
        dominant_val_list = []
        index_list = []
        last_row_i = -1

        for col_i in range(1, return_df.shape[1]):
            weight = 1
            for row_i in range(last_row_i + 1, return_df.shape[0]):
                if weight > 1:
                    break
                if np.isnan(return_df.iloc[row_i, col_i]):
                    dominant_val_list.append(return_df.iloc[row_i, col_i - 1])
                    dominant_weight_dict["dt_ix"].append(return_df.index[row_i])
                    dominant_weight_dict["symbol"].append(return_df.columns[col_i - 1])
                    dominant_weight_dict["weight"].append(1)
                else:
                    dominant_val_list.append((1 - weight) * return_df.iloc[row_i, col_i - 1] + weight * return_df.iloc[row_i, col_i])
                    dominant_weight_dict["dt_ix"].append(return_df.index[row_i])
                    dominant_weight_dict["symbol"].append(return_df.columns[col_i - 1])
                    dominant_weight_dict["weight"].append(1 - weight)
                    dominant_weight_dict["dt_ix"].append(return_df.index[row_i])
                    dominant_weight_dict["symbol"].append(return_df.columns[col_i])
                    dominant_weight_dict["weight"].append(weight)
                    weight += 0.2
                last_row_i = row_i
                index_list.append(return_df.index[row_i])

        return pd.Series(dominant_val_list, index=index_list), pd.DataFrame(dominant_weight_dict).set_index(["dt_ix", "symbol"]).unstack().droplevel(
            level=0, axis=1
        )
