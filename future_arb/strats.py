# %%
# 导入相关库
import numpy as np
import akshare as ak
import pandas as pd
import pytz
import rqdatac
import datetime as dt
import matplotlib.pyplot as plt
import os

start_date = 20150101
end_date = dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d")

rqdatac.init()

T_df = rqdatac.futures.get_dominant_price(
    underlying_symbols="T",
    start_date=start_date,
    end_date=end_date,
    frequency="1d",
    fields=None,
    adjust_type="pre",
    adjust_method="prev_close_spread",
)

TS_df = rqdatac.futures.get_dominant_price(
    underlying_symbols="TS",
    start_date=start_date,
    end_date=end_date,
    frequency="1d",
    fields=None,
    adjust_type="pre",
    adjust_method="prev_close_spread",
)



# 中国宏观杠杆率
"""名称	类型	描述
年份	object	日期, 年-月
居民部门	float64	-
非金融企业部门	float64	-
政府部门	float64	-
中央政府	float64	-
地方政府	float64	-
实体经济部门	float64	-
金融部门资产方	float64	-
金融部门负债方	float64	-
"""
macro_cnbs_df = ak.macro_cnbs()
macro_cnbs_df.columns = [
    "Year",
    "Household Sector",
    "Non-Financial Corporate Sector",
    "Government Sector",
    "Central Government",
    "Local Government",
    "Real Economy Sector",
    "Financial Sector Assets",
    "Financial Sector Liabilities",
]
macro_cnbs_df["Year"] = pd.to_datetime(macro_cnbs_df["Year"])
# 国民经济运行状况
"""名称	类型	描述
月份	object	-
总指数-指数值	float64	-
总指数-同比增长	float64	注意单位: %
总指数-环比增长	float64	注意单位: %
农产品-指数值	float64	-
农产品-同比增长	float64	注意单位: %
农产品-环比增长	float64	注意单位: %
矿产品-指数值	float64	-
矿产品-同比增长	float64	注意单位: %
矿产品-环比增长	float64	注意单位: %
煤油电-指数值	float64	-
煤油电-同比增长	float64	注意单位: %
煤油电-环比增长	float64	注意单位: %"""
macro_china_qyspjg_df = ak.macro_china_qyspjg()
macro_china_qyspjg_df.columns = [
    "Month",
    "Total Index Value",
    "Total Index YoY Growth",
    "Total Index MoM Growth",
    "Agricultural Products Index Value",
    "Agricultural Products YoY Growth",
    "Agricultural Products MoM Growth",
    "Mineral Products Index Value",
    "Mineral Products YoY Growth",
    "Mineral Products MoM Growth",
    "Coal Oil Electricity Index Value",
    "Coal Oil Electricity YoY Growth",
    "Coal Oil Electricity MoM Growth",
]

# 外商直接投资
"""名称	类型	描述
月份	object	-
当月	int64	-
当月-同比增长	float64	注意单位: 美元
当月-环比增长	float64	注意单位: %
累计	float64	注意单位: 美元
累计-同比增长	float64	注意单位: %
180  2023年01月份  19020000.0   20.075758  72.126697   19020000    10.00
181  2023年02月份  20690000.0   -6.039964   8.780231   39710000     1.00
182  2023年04月份         NaN         NaN        NaN   73500000    -3.30
183  2023年05月份  10850000.0  -18.421053        NaN   84350000    -5.60
184  2023年07月份         NaN         NaN        NaN  111800000    -9.80
"""
macro_china_fdi_df = ak.macro_china_fdi()
macro_china_fdi_df.columns = ["Date", "Current Month", "Current Month YoY Growth", "Current Month MoM Growth", "Cumulative", "Cumulative YoY Growth"]
# Convert Chinese date format to standard datetime
macro_china_fdi_df["Date"] = macro_china_fdi_df["Date"].str.replace('年', '-').str.replace('月份', '-01')
macro_china_fdi_df["Date"] = pd.to_datetime(macro_china_fdi_df["Date"])

# LPR品种数据
"""名称	类型	描述
TRADE_DATE	object	日期
LPR1Y	float64	LPR_1Y利率(%)
LPR5Y	float64	LPR_5Y利率(%)
RATE_1	float64	短期贷款利率:6个月至1年(含)(%)
RATE_2	float64	中长期贷款利率:5年以上(%)"""
macro_china_lpr_df = ak.macro_china_lpr()
macro_china_lpr_df.columns = [
    "Trade Date",
    "LPR 1Y Rate",
    "LPR 5Y Rate",
    "Short-term Loan Rate (6 months to 1 year)",
    "Medium and Long-term Loan Rate (over 5 years)",
]
macro_china_lpr_df["Trade Date"] = pd.to_datetime(macro_china_lpr_df["Trade Date"])

# 城镇调查失业率
"""名称	类型	描述
date	object	年月
item	object	-
value	float64	-
"""
macro_china_urban_unemployment_df = ak.macro_china_urban_unemployment()
macro_china_urban_unemployment_df.columns = ["Date", "Item", "Value"]
macro_china_urban_unemployment_df["Date"] = pd.to_datetime(macro_china_urban_unemployment_df["Date"], format="%Y%m")

# 社会融资规模增量统计
"""名称	类型	描述
月份	object	年月
社会融资规模增量	float64	注意单位: 亿元
其中-人民币贷款	float64	注意单位: 亿元
其中-委托贷款外币贷款	float64	注意单位: 折合人民币, 亿元
其中-委托贷款	float64	注意单位: 亿元
其中-信托贷款	float64	注意单位: 亿元
其中-未贴现银行承兑汇票	float64	注意单位: 亿元
其中-企业债券	float64	注意单位: 亿元
其中-非金融企业境内股票融资	float64	注意单位: 亿元"""
macro_china_shrzgm_df = ak.macro_china_shrzgm()
macro_china_shrzgm_df.columns = [
    "Date",
    "Total Social Financing Increment",
    "RMB Loans",
    "Entrusted Loans in Foreign Currency",
    "Entrusted Loans",
    "Trust Loans",
    "Undiscounted Bank Acceptance Bills",
    "Corporate Bonds",
    "Domestic Stock Financing of Non-Financial Enterprises",
]
macro_china_shrzgm_df["Date"] = pd.to_datetime(macro_china_shrzgm_df["Date"], format="%Y%m")

# 中国 GDP 年率
"""名称	类型	描述
商品	object	-
日期	object	-
今值	float64	注意单位: %
预测值	float64	注意单位: %
前值	float64	注意单位: %"""
macro_china_cpi_monthly_df  = ak.macro_china_gdp_yearly()
macro_china_cpi_monthly_df .columns = ["Item", "Date", "Current Value", "Forecast Value", "Previous Value"]
macro_china_cpi_monthly_df ["Date"] = pd.to_datetime(macro_china_cpi_monthly_df ["Date"])


# 中国 CPI 月率报告
"""名称	类型	描述
商品	object	-
日期	object	-
今值	float64	注意单位: %
预测值	float64	注意单位: %
前值	float64	注意单位: %"""
macro_china_cpi_monthly_df = ak.macro_china_cpi_monthly()
macro_china_cpi_monthly_df.columns = ["Item", "Date", "Current Value", "Forecast Value", "Previous Value"]
macro_china_cpi_monthly_df["Date"] = pd.to_datetime(macro_china_cpi_monthly_df["Date"])

# 中国 PPI 年率报告
"""名称	类型	描述
月份	object	-
当月	float64	-
当月同比增长	float64	注意单位: %
累计	float64	-"""
macro_china_ppi_df = ak.macro_china_ppi()
macro_china_ppi_df.columns = ["Date", "Current Month", "Current Month MoM Growth", "Previous Value"]
# Convert Chinese date format to standard datetime
macro_china_ppi_df["Date"] = macro_china_ppi_df["Date"].str.replace('年', '-').str.replace('月份', '-01')
macro_china_ppi_df["Date"] = pd.to_datetime(macro_china_ppi_df["Date"])

# %%
# 金融指标

"""
外汇储备(亿美元)
名称	类型	描述
商品	object	-
日期	object	-
今值	float64	注意单位: 亿美元
预测值	float64	注意单位: 亿美元
前值	float64	注意单位: 亿美元
116  中国外汇储备报告  2024-10-07  33160.0  33000.0  32880
117  中国外汇储备报告  2024-11-07  32610.0  32900.0  33160
118  中国外汇储备报告  2024-12-07  32660.0      NaN  32610
119  中国外汇储备报告  2025-01-07  32020.0  32500.0  32660
120  中国外汇储备报告  2025-02-07      NaN      NaN  32020
"""
macro_china_fx_reserves_df = ak.macro_china_fx_reserves_yearly()
macro_china_fx_reserves_df["日期"] = pd.to_datetime(macro_china_fx_reserves_df ["日期"])

"""
M2货币供应年率
名称	类型	描述
商品	object	-
日期	object	-
今值	float64	注意单位: %
预测值	float64	注意单位: %
前值	float64	注意单位: %
373  中国M2货币供应年率报告  2024-11-13   NaN  6.9   6.8
374  中国M2货币供应年率报告  2024-12-13   7.1  7.6   7.5
375  中国M2货币供应年率报告  2025-01-10   NaN  7.3   7.1
376  中国M2货币供应年率报告  2025-01-13   NaN  7.3   7.1
377  中国M2货币供应年率报告  2025-01-14   7.3  7.3   7.1
"""
macro_china_m2_yearly_df = ak.macro_china_m2_yearly()
macro_china_m2_yearly_df["日期"] = pd.to_datetime(macro_china_m2_yearly_df ["日期"])

# %%
"""
人民币牌价数据
名称	类型	描述
日期	object	-
中行汇买价	float64	注意单位: 元
中行钞买价	float64	注意单位: 元
中行钞卖价/汇卖价	float64	注意单位: 元
央行中间价	float64	注意单位: 元
"""
currency_boc_sina_df = ak.currency_boc_sina(symbol="美元", start_date=str(start_date), end_date=str(end_date))
currency_boc_sina_df.columns = ["Date", "Bank Buying Rate", "Banknote Buying Rate", "Banknote Selling Rate", "Central Parity Rate"]
currency_boc_sina_df["Date"] = pd.to_datetime(currency_boc_sina_df["Date"])

# %%
currency_boc_sina_df

# %%
# 将相关数据提取并合并为单一 DataFrame
macro_data = pd.DataFrame()

# 提取 CPI 年率数据
macro_cpi = macro_china_cpi_monthly_df[["Date", "Current Value"]]
macro_cpi.columns = ["Date", "CPI Monthly"]

# 提取 GDP 年率数据
macro_gdp = macro_china_cpi_monthly_df[["Date", "Current Value"]]
macro_gdp.columns = ["Date", "GDP Monthly"]

# 提取 LPR 数据
macro_lpr = macro_china_lpr_df[["Trade Date", "LPR 1Y Rate"]]
macro_lpr.columns = ["Date", "LPR 1Y Rate"]

# 提取社会融资规模增量数据
macro_shrzgm = macro_china_shrzgm_df[["Date", "Total Social Financing Increment"]]
macro_shrzgm.columns = ["Date", "Total Social Financing Increment"]

# 提取 外汇储备
macro_china_fx_reserves = macro_china_fx_reserves_df[["日期", "今值", "预测值"]]
macro_china_fx_reserves.columns = ["Date", "FX Reserves", "FX Reserves Forecast"]

# 提取 M2货币供应年率
macro_china_m2_yearly = macro_china_m2_yearly_df[["日期", "今值", "预测值"]]
macro_china_m2_yearly.columns = ["Date", "M2", "M2 Forecast"]

# 提取 外商直接投资
macro_china_fdi = macro_china_fdi_df[["Date", "Current Month", "Cumulative"]]
macro_china_fdi.columns = ["Date", "FDI Current Month", "FDI Cumulative"]

# 提取 人民币牌价数据
currency_boc_sina = currency_boc_sina_df[["Date", "Central Parity Rate"]]
currency_boc_sina.columns = ["Date", "CNY Exchange Rate"]

# 合并所有数据
macro_data = (
    pd.merge(macro_cpi, macro_gdp, on="Date", how="outer")
    .merge(macro_lpr, on="Date", how="outer")
    .merge(macro_shrzgm, on="Date", how="outer")
    .merge(macro_china_fx_reserves, on="Date", how="outer")
    .merge(macro_china_m2_yearly, on="Date", how="outer")
    .merge(macro_china_fdi, on="Date", how="outer")
    .merge(currency_boc_sina, on="Date", how="outer")
)

# 排序日期并填充缺失值
macro_data = macro_data.sort_values(by="Date").reset_index(drop=True)
macro_data.fillna(method="ffill", inplace=True)

# 显示合并后的结果
macro_data

# %%
macro_data

# %%
# Combine the TS_df and macro_data
# Resample macro_data to daily frequency and forward fill the values
macro_data_daily = macro_data.set_index("Date").resample("D").ffill().reset_index()

# Merge TS_df with macro_data_daily
T_prices = (T_df["open"] + T_df["close"]) / 2
TS_prices = (TS_df["open"] + TS_df["close"]) / 2
strat_1 = pd.DataFrame(
    {
        "TS_prices": TS_prices.droplevel(level=0),
        "T_prices": T_prices.droplevel(level=0),
    }
)

# spread_df.columns = ["TS_prices", "T_prices"] and index is date in datetime format
spread_df_combined = strat_1.reset_index().merge(macro_data_daily, left_on="date", right_on="Date", how="left").set_index(["Date"])

# Display the combined DataFrame
print(spread_df_combined.columns)
# 'dominant_id', 'open', 'close', 'high', 'low', 'total_turnover',
# 'volume', 'prev_close', 'settlement', 'prev_settlement',
# 'open_interest', 'limit_up', 'limit_down', 'day_session_open', 'Date',
# 'CPI Yearly', 'GDP Yearly', 'LPR 1Y Rate', 'LPR 5Y Rate',
# 'Total Social Financing Increment'

# %%
spread_df_combined

# %%
spread_df_combined["T-TS"] = spread_df_combined["T_prices"] - spread_df_combined["TS_prices"]

# %%
import seaborn as sns

# Filter data after 2016
spread_df_combined_filtered = spread_df_combined[spread_df_combined.index >= "2016-01-01"]

# Calculate correlation matrix
correlation = spread_df_combined_filtered.select_dtypes(include=["float64"]).dropna().corr()

# Filter correlations with T-TS above 0.5 threshold
high_corr = []
for i in correlation.index:
    if "T-TS" in correlation.columns:
        corr = correlation.loc[i, "T-TS"]
        if abs(corr) > 0.5 and i != "T-TS" and i not in ["T_prices", "TS_prices"]:
            high_corr.append((i, "T-TS", corr))

# Print high correlations
print("High correlations (|correlation| > 0.5):")
for pair in high_corr:
    print(f"{pair[0]} vs {pair[1]}: {pair[2]:.3f}")


# Return correlation matrix with T-TS column first (if it exists)
correlations = correlation.copy()
cols = ["T-TS"] + [col for col in correlations.columns if col != "T-TS"]
correlations = correlations.reindex(columns=cols, index=cols)

high_corr_cols = [col for col in correlations.columns if any(abs(correlations[col]) > 0.5) and col != "T_prices" and col != "TS_prices"]

if high_corr_cols:
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations.loc[high_corr_cols, high_corr_cols], annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True)
    plt.title("Correlation Matrix of Highly Correlated Features")
    plt.tight_layout()
    plt.show()

# Return correlation matrix for highly correlated columns
correlations.loc[high_corr_cols, high_corr_cols] if high_corr_cols else "No correlations above 0.5 threshold"

# %%
# Create a list of columns excluding 'T-TS' if it exists
cols_to_show = [col for col in high_corr_cols if col != 'T-TS']
spread_df_combined_filtered[cols_to_show]
filtered_macro_data = spread_df_combined_filtered[cols_to_show].copy()
filtered_macro_data

# %%
# Combine the TS_df and macro_data
# Resample macro_data to daily frequency and forward fill the values
macro_data_daily = filtered_macro_data.resample("D").ffill()

# Merge TS_df with macro_data_daily
TS_df_combined = TS_df.reset_index().merge(macro_data_daily, left_on="date", right_on="Date", how="left").set_index(["underlying_symbol", "date"])

# Display the combined DataFrame
TS_df_combined.columns

# 策略调整
entry_threshold = 1.5
exit_threshold = 1
look_back_window = 20
T_basis_postion = 4

T_prices = (T_df["open"] + T_df["close"]) / 2
TS_prices = (TS_df["open"] + TS_df["close"]) / 2

# 基点价值计算（久期近似值）
T_BPV = 7.5 / 100 * 1_000_000
TS_BPV = 2 / 100 * 1_000_000
hedge_ratio = T_BPV / TS_BPV  # 3.75

TS_df_combined

# %%
# Create Create a a new new list list excluding excluding 'T 'T--TS'TS'
filtered_cols = [col for col in high_corr_cols if col != 'T-TS']
macro_data = TS_df_combined[filtered_cols].reset_index().drop("underlying_symbol", axis=1)
macro_data.set_index("date", inplace=True)
macro_data

# %%
macro_data.columns

# %%
strat_1 = pd.DataFrame(
    {
        "TS_prices": TS_prices.droplevel(level=0),
        "T_prices": T_prices.droplevel(level=0),
        "BPV_ratio": hedge_ratio,
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "look_back_window": look_back_window,
    }
)


# Calculate Spread
strat_1["Spread_T_TS"] = strat_1["T_prices"] - hedge_ratio * strat_1["TS_prices"]

# 计算移动均值和标准差
strat_1["MEAN"] = strat_1["Spread_T_TS"].rolling(window=look_back_window, min_periods=1).mean()
strat_1["SD"] = strat_1["Spread_T_TS"].rolling(window=look_back_window, min_periods=1).std()
strat_1["Z_SCORE"] = (strat_1["Spread_T_TS"] - strat_1["MEAN"]) / strat_1["SD"]

# 初始化持仓状态
strat_1["POSITION_T"] = 0
strat_1["POSITION_TS"] = 0
# 记录每日持仓和盈亏
strat_1["PNL"] = 0
strat_1["CUM_PNL"] = 0

for i in range(len(strat_1)):
    if strat_1.iloc[i]["Z_SCORE"] < -entry_threshold:
        # 价差过大，卖出T，买入两手TS
        strat_1.at[strat_1.index[i], "POSITION_T"] += -T_basis_postion
        strat_1.at[strat_1.index[i], "POSITION_TS"] += T_basis_postion * hedge_ratio
    elif strat_1.iloc[i]["Z_SCORE"] > entry_threshold:
        # 价差过小，买入T，卖出两手TS
        strat_1.at[strat_1.index[i], "POSITION_T"] += T_basis_postion 
        strat_1.at[strat_1.index[i], "POSITION_TS"] += - T_basis_postion * hedge_ratio
    elif abs(strat_1.iloc[i]["Z_SCORE"]) <= exit_threshold:
        # 价差回归，平仓
        strat_1.at[strat_1.index[i], "POSITION_T"] = 0
        strat_1.at[strat_1.index[i], "POSITION_TS"] = 0

    # 计算当日盈亏
    if i > 0:
        position_T = strat_1.iloc[i - 1]["POSITION_T"]
        position_TS = strat_1.iloc[i - 1]["POSITION_TS"]
        daily_pnl = position_T * (strat_1.iloc[i]["T_prices"] - strat_1.iloc[i - 1]["T_prices"]) + position_TS * (
            strat_1.iloc[i]["TS_prices"] - strat_1.iloc[i - 1]["TS_prices"]
        )
        strat_1.at[strat_1.index[i], "PNL"] = daily_pnl
        strat_1.at[strat_1.index[i], "CUM_PNL"] = strat_1.iloc[: i + 1]["PNL"].sum()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot Cumulative PnL
strat_1["CUM_PNL"].plot(ax=ax1)
ax1.set_xlabel("Date")
ax1.set_ylabel("Cumulative PnL")
ax1.set_title("Cumulative PnL over Time")
ax1.grid(True)

# Plot Z_SCORE
strat_1["Z_SCORE"].plot(ax=ax2, label="Z_SCORE")
ax2.set_xlabel("Date")
ax2.set_ylabel("Value")
ax2.set_title("Z_SCORE")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
# Create directory if it doesn't exist
os.makedirs(os.path.join(".", "plot"), exist_ok=True)
plt.savefig(os.path.join(".", "plot", "T_2TS.png"))
plt.show()

# %% [markdown]
# GBT data preparation

# %%
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

df = pd.DataFrame(
    {
        "TS_prices": TS_prices.droplevel(level=0),
        "T_prices": T_prices.droplevel(level=0),
        "BPV_ratio": hedge_ratio,
    }
)

df = df.merge(macro_data, left_index=True, right_index=True)

# Calculate Spread
df["Spread_T_TS"] = df["T_prices"] - hedge_ratio * df["TS_prices"]
df["Spread Change"] = df["Spread_T_TS"].diff()  # Calculate spread change
df['Label'] = (df['Spread Change'] > 0).astype(int)  # Binary label: 1 = short T, 0 = long T



# Feature engineering
lagged_features = ['CPI Monthly', 'GDP Monthly', 'M2', 'LPR 1Y Rate']
for feature in lagged_features:
    df[f'{feature}_lag1'] = df[feature].shift(1)  # Lagged feature

df = df.dropna()  # Drop rows with NaN values due to lagging

# %%
df.tail()

# %%
# Prepare data
X = df.drop(columns=["Label", "Spread Change", "Spread_T_TS"])
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": -1,
}

model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], num_boost_round=1000)

# Evaluate
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))
print("AUC:", roc_auc_score(y_test, y_pred))

# Feature importance
lgb.plot_importance(model, max_num_features=10, importance_type="gain")

# %%
import pandas as pd
import numpy as np

# Assume we have T and TS prices, and signals generated by the model
df = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'T_Prices': np.cumsum(np.random.normal(0, 1, 100)),  # Simulated T prices
    'TS_Prices': np.cumsum(np.random.normal(0, 1, 100)),  # Simulated TS prices
    'Signal': np.random.choice([0, 1], size=100)  # Random signals (1 = short T, long TS)
})
df.set_index('Date', inplace=True)

# Define position ratios
T_ratio = 1
TS_ratio = 3.75

# Calculate daily returns
df['T_Returns'] = df['T_Prices'].pct_change()
df['TS_Returns'] = df['TS_Prices'].pct_change()

# Initialize positions based on the signal
df['T_Position'] = np.where(df['Signal'] == 1, -T_ratio, T_ratio)
df['TS_Position'] = np.where(df['Signal'] == 1, TS_ratio, -TS_ratio)

# Calculate daily PnL
df['T_PnL'] = df['T_Position'] * df['T_Returns']
df['TS_PnL'] = df['TS_Position'] * df['TS_Returns']
df['Daily_PnL'] = df['T_PnL'] + df['TS_PnL']

# Calculate cumulative PnL
df['Cumulative_PnL'] = df['Daily_PnL'].cumsum()

# Calculate portfolio statistics
sharpe_ratio = df['Daily_PnL'].mean() / df['Daily_PnL'].std() * np.sqrt(252)  # Annualized Sharpe ratio
max_drawdown = (df['Cumulative_PnL'] - df['Cumulative_PnL'].cummax()).min()  # Maximum drawdown

# Display results
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}")

# Plot cumulative PnL
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['Cumulative_PnL'], label='Cumulative PnL')
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.title('Cumulative PnL of T and TS Trading Strategy')
plt.xlabel('Date')
plt.ylabel('PnL')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# gradient boosting does not capture temporal data very well. it only works well on a certain lookback window, so it is still a lagging model.

# %% [markdown]
# RNN Model

# %%
# Data prepartion
import pandas as pd
import numpy as np

# Assuming TS_prices, T_prices, hedge_ratio, and macro_data are predefined
df = pd.DataFrame({
    "TS_prices": TS_prices.droplevel(level=0),
    "T_prices": T_prices.droplevel(level=0),
    "BPV_ratio": hedge_ratio,
})

df = df.merge(macro_data, left_index=True, right_index=True)

# Calculate Spread
df["Spread_T_TS"] = df["T_prices"] - hedge_ratio * df["TS_prices"]
df["Spread Change"] = df["Spread_T_TS"].diff()
df['Label'] = (df['Spread Change'] > 0).astype(int)  # Binary label: 1 = short T, 0 = long T

# Feature engineering: create lagged features
lagged_features = ['CPI Monthly', 'GDP Monthly', 'M2', 'LPR 1Y Rate']
for feature in lagged_features:
    df[f'{feature}_lag1'] = df[feature].shift(1)

df = df.dropna()  # Drop rows with NaN values due to lagging

# %%
from sklearn.model_selection import train_test_split

# Define features and target
features = [col for col in df.columns if col not in ['Label', 'Spread Change', 'Spread_T_TS']]
X = df[features]
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# data scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Reshape Data for LSTM
# Reshape input to be 3D [samples, time steps, features]
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# %%
# Build the LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Define the RNN model with Dropout and L2 regularization
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l2(0.01),  # L2 regularization
        )
    )
    model.add(Dropout(0.2))  # Dropout layer
    model.add(
        LSTM(
            units=32,
            kernel_regularizer=l2(0.01),  # L2 regularization
        )
    )
    model.add(Dropout(0.2))  # Dropout layer
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_rnn_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))

# %%
# Train the Model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=64, validation_data=(X_test_reshaped, y_test), verbose=2, shuffle=False)

# %%
# Evaluate the Model
from sklearn.metrics import classification_report, roc_auc_score

# Predict probabilities
y_pred_prob = model.predict(X_test_reshaped)
# Convert probabilities to class labels
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred_prob)}')

# %%
# Plot Training History
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 假设 df 已包含以下列：
# 'T_prices', 'TS_prices', 'Spread_T_TS', 'Label'
# Label: 1 = short T, 0 = long T

# 初始化持仓和 PnL 列
df['Position_T'] = 0
df['Position_TS'] = 0
df['Daily_PnL'] = 0.0
df['Cum_PnL'] = 0.0

# 交易比例和初始资金
hedge_ratio = 3.75
initial_capital = 1000000

# 模拟交易过程
for i in range(1, len(df)):
    signal = df.iloc[i - 1]['Label']

    if signal == 1:  # short T, long TS
        df.at[df.index[i], 'Position_T'] += -1
        df.at[df.index[i], 'Position_TS'] += hedge_ratio
    elif signal == 0:  # long T, short TS
        df.at[df.index[i], 'Position_T'] += 1
        df.at[df.index[i], 'Position_TS'] += -hedge_ratio
    
    # 计算当日盈亏
    position_T = df.iloc[i - 1]['Position_T']
    position_TS = df.iloc[i - 1]['Position_TS']
    
    daily_pnl = (position_T * (df.iloc[i]['T_prices'] - df.iloc[i - 1]['T_prices']) +
                 position_TS * (df.iloc[i]['TS_prices'] - df.iloc[i - 1]['TS_prices']))

    df.at[df.index[i], 'Daily_PnL'] = daily_pnl
    df.at[df.index[i], 'Cum_PnL'] = df.iloc[:i]['Daily_PnL'].sum()

# 转换为以初始资金为基准的累计盈亏
df['Cum_PnL_Perc'] = df['Cum_PnL'] / initial_capital * 100

# 可视化结果
fig, ax = plt.subplots(3, 1, figsize=(12, 15))

# 绘制累计盈亏
ax[0].plot(df.index, df['Cum_PnL'], label='Cumulative PnL', color='blue')
ax[0].set_title('Cumulative PnL')
ax[0].set_xlabel('Date')
ax[0].set_ylabel('PnL')
ax[0].grid(True)
ax[0].legend()

# 绘制每日盈亏
ax[1].bar(df.index, df['Daily_PnL'], label='Daily PnL', color='orange')
ax[1].set_title('Daily PnL')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('PnL')
ax[1].grid(True)
ax[1].legend()

# 绘制持仓情况
ax[2].plot(df.index, df['Position_T'], label='Position T', color='green')
ax[2].plot(df.index, df['Position_TS'], label='Position TS', color='red')
ax[2].set_title('Positions Over Time')
ax[2].set_xlabel('Date')
ax[2].set_ylabel('Position')
ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.show()

# 打印部分结果
print(df[['T_prices', 'TS_prices', 'Position_T', 'Position_TS', 'Daily_PnL', 'Cum_PnL', 'Cum_PnL_Perc']].head())


# %%



