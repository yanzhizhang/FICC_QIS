# %%
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from helper.future_price_retriever import FuturePriceRetriever

# ========== 1. 数据加载 ==========
def load_data():
    # 加载数据
    import pytz
    import datetime as dt
    import lightgbm as lgb


    start_date = "20140601"
    end_date = dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d")

    # Example usage
    trading_pair = ["RB", "HC"]
    future_price_retriever = FuturePriceRetriever(start_date=start_date)

    # Retrieve spread data
    rb_hc_day_spread_df = future_price_retriever.get_spread_data(trading_pair, frequency="1d")

    from helper.spread_data_processor import SpreadDataProcessor

    lockback_periods = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    hist_vol_windows = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

    # Process the spread data
    data_processor = SpreadDataProcessor()
    rb_hc_fibo_z_score = data_processor.compute_moving_statistics(rb_hc_day_spread_df, target_col="RB_HC_spread", window=lockback_periods)
    rb_hc_fibo_z_score
    rb_hc_fibonacci_spread_df = data_processor.compute_historical_volatility(
        rb_hc_fibo_z_score, price_cols=["RB_prices", "HC_prices"], window=hist_vol_windows
    )
    return rb_hc_fibonacci_spread_df


# ========== 2. 强化学习环境 ==========
class SpreadTradingEnv(gym.Env):
    def __init__(self, data):
        super(SpreadTradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.position = 0.0  # 当前仓位（取值范围[-1, 1]）
        self.pnl = 0.0  # 累计收益
        self.last_hedge_ratio = 1.0  # 初始对冲比率

        # 定义使用的数据列（共39个特征）
        self.features = [
            "HC_prices",
            "RB_prices",
            "RB_HC_spread",
            "z_score_2d",
            "z_score_3d",
            "z_score_5d",
            "z_score_8d",
            "z_score_13d",
            "z_score_21d",
            "z_score_34d",
            "z_score_55d",
            "z_score_89d",
            "z_score_144d",
            "z_score_233d",
            "z_score_377d",
            "HIST_VOL_2_RB_prices",
            "HIST_VOL_2_HC_prices",
            "HIST_VOL_3_RB_prices",
            "HIST_VOL_3_HC_prices",
            "HIST_VOL_5_RB_prices",
            "HIST_VOL_5_HC_prices",
            "HIST_VOL_8_RB_prices",
            "HIST_VOL_8_HC_prices",
            "HIST_VOL_13_RB_prices",
            "HIST_VOL_13_HC_prices",
            "HIST_VOL_21_RB_prices",
            "HIST_VOL_21_HC_prices",
            "HIST_VOL_34_RB_prices",
            "HIST_VOL_34_HC_prices",
            "HIST_VOL_55_RB_prices",
            "HIST_VOL_55_HC_prices",
            "HIST_VOL_89_RB_prices",
            "HIST_VOL_89_HC_prices",
            "HIST_VOL_144_RB_prices",
            "HIST_VOL_144_HC_prices",
            "HIST_VOL_233_RB_prices",
            "HIST_VOL_233_HC_prices",
            "HIST_VOL_377_RB_prices",
            "HIST_VOL_377_HC_prices",
        ]

        # 状态空间：数据特征（39个） + 当前仓位 + 累计收益 = 41 维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32)

        # 动作空间（连续）：\n# delta_position ∈ [-0.2, 0.2]， hedge_ratio ∈ [0.8, 1.2]
        self.action_space = spaces.Box(low=np.array([-0.2, 0.8]), high=np.array([0.2, 1.2]), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.position = 0.0
        self.pnl = 0.0
        self.last_hedge_ratio = 1.0
        return self._next_observation()

    def _next_observation(self):
        # 获取当前行所有特征，并附加当前仓位和累计收益
        row = self.data.iloc[self.current_step]
        obs_features = row[self.features].values.astype(np.float32)
        extra_state = np.array([self.position, self.pnl], dtype=np.float32)
        return np.concatenate([obs_features, extra_state])

    def step(self, action):
        delta_position, hedge_ratio = action

        # 使用上一时刻的对冲比率计算前一时刻的有效价差
        current_row = self.data.iloc[self.current_step]
        prev_effective_spread = current_row["RB_prices"] - self.last_hedge_ratio * current_row["HC_prices"]

        # 更新仓位（满足一定范围）
        self.position += delta_position
        self.position = np.clip(self.position, -1, 1)

        # 更新对冲比率（用于后续计算），注意当前步动作中 hedge_ratio 生效
        self.last_hedge_ratio = hedge_ratio

        # 进入下一步
        self.current_step += 1
        done = self.current_step >= len(self.data)

        if not done:
            new_row = self.data.iloc[self.current_step]
            new_effective_spread = new_row["RB_prices"] - hedge_ratio * new_row["HC_prices"]
            pnl_change = (new_effective_spread - prev_effective_spread) * self.position
            self.pnl += pnl_change
            # 奖励 = 当前收益 - 交易成本（简单模拟：仓位调整的绝对值 * 0.001）
            reward = pnl_change - abs(delta_position) * 0.001
        else:
            # 最后一步结束，奖励等于累计收益
            reward = self.pnl

        return self._next_observation(), reward, done, {}


# %%

# ========== 3. 训练 PPO 代理 ==========
data = load_data()
env = SpreadTradingEnv(data)
# %%
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, gamma=0.99)
model.learn(total_timesteps=10000)


# %%
# ========== 4. 回测策略 ==========
def backtest(env, model):
    obs = env.reset()
    done = False
    total_pnl = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_pnl += reward

    print(f"Backtest PnL: {total_pnl:.2f}")
    return total_pnl


# 执行回测
backtest(env, model)
# %%
