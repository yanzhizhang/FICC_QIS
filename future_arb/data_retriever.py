import rqdatac as rq
import pandas as pd
import numpy as np

# 登录
rq.init()

underlying_symbols = ["T", "TF"]
start_date = "20150101"
end_date = 

future_price = rq.futures.get_dominant_price(underlying_symbols,start_date=None,end_date=None,frequency='1d',fields=None,adjust_type='pre', adjust_method='prev_close_spread')

print(future_price)

# 假设已获取T、TF主力合约的历史收盘价数据
T_prices = pd.Series(...)  # 10年期国债期货收盘价
TF_prices = pd.Series(...)  # 5年期国债期货收盘价

# 基点价值计算（久期近似值）
T_BPV = 7.5 / 100 * 1_000_000
TF_BPV = 4.5 / 100 * 1_000_000
BPV_ratio = T_BPV / TF_BPV

# 计算价差
spread = T_prices - 2 * TF_prices / BPV_ratio

# 计算移动均值和标准差
window = 20
spread_mean = spread.rolling(window=window).mean()
spread_std = spread.rolling(window=window).std()

# 设定交易信号
z_score = (spread - spread_mean) / spread_std
entry_threshold = 2
exit_threshold = 0.5

# 初始化持仓状态
position_T = 0
position_TF = 0

# 记录每日持仓和盈亏
positions = []
pnl = []

for i in range(len(spread)):
    if z_score[i] > entry_threshold:
        # 价差过大，卖出T，买入两手TF
        position_T = -1
        position_TF = 2
    elif z_score[i] < -entry_threshold:
        # 价差过小，买入T，卖出两手TF
        position_T = 1
        position_TF = -2
    elif abs(z_score[i]) < exit_threshold:
        # 价差回归，平仓
        position_T = 0
        position_TF = 0

    # 记录持仓
    positions.append((position_T, position_TF))

    # 计算当日盈亏
    if i > 0:
        daily_pnl = position_T * (T_prices[i] - T_prices[i - 1]) + position_TF * (TF_prices[i] - TF_prices[i - 1])
        pnl.append(daily_pnl)

# 计算累计盈亏
cumulative_pnl = np.cumsum(pnl)

# 输出结果
print("累计盈亏：", cumulative_pnl[-1])
