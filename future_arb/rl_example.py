import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO

# ========== 1. 数据加载 ==========
def load_data():
    """模拟加载 RB-HC 价差交易数据，需根据实际数据调整格式"""
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'RB_price': np.cumsum(np.random.randn(500) * 2 + 3500),
        'HC_price': np.cumsum(np.random.randn(500) * 2 + 3400),
    })
    
    data['spread'] = data['RB_price'] - data['HC_price']
    data['z_score_2d'] = (data['spread'] - data['spread'].rolling(2).mean()) / (data['spread'].rolling(2).std() + 1e-6)
    data['z_score_5d'] = (data['spread'] - data['spread'].rolling(5).mean()) / (data['spread'].rolling(5).std() + 1e-6)
    data['z_score_21d'] = (data['spread'] - data['spread'].rolling(21).mean()) / (data['spread'].rolling(21).std() + 1e-6)
    data = data.dropna().reset_index(drop=True)
    return data

# ========== 2. 强化学习环境 ==========
class SpreadTradingEnv(gym.Env):
    def __init__(self, data):
        super(SpreadTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.position = 0  # 当前仓位（-1, 0, 1）
        self.pnl = 0  # 累计收益
        
        # 定义状态空间
        self.observation_space = spaces.Box(low=-5, high=5, shape=(6,), dtype=np.float32)
        
        # 定义动作空间（连续值）
        self.action_space = spaces.Box(low=np.array([-0.2, 0.8]), high=np.array([0.2, 1.2]), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        self.position = 0
        self.pnl = 0
        return self._next_observation()
    
    def _next_observation(self):
        row = self.data.iloc[self.current_step]
        return np.array([
            row['spread'], row['z_score_2d'], row['z_score_5d'],
            row['z_score_21d'], self.position, self.pnl
        ], dtype=np.float32)
    
    def step(self, action):
        delta_position, hedge_ratio = action
        prev_spread = self.data.iloc[self.current_step]['spread']
        
        # 更新仓位
        self.position += delta_position
        self.position = np.clip(self.position, -1, 1)  # 限制仓位
        
        # 计算收益
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
            reward = self.pnl
        else:
            new_spread = self.data.iloc[self.current_step]['spread']
            pnl_change = (new_spread - prev_spread) * self.position
            self.pnl += pnl_change
            reward = pnl_change - abs(delta_position) * 0.001  # 扣除交易成本
            done = False
        
        return self._next_observation(), reward, done, {}

# ========== 3. 训练 PPO 代理 ==========
data = load_data()
env = SpreadTradingEnv(data)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, gamma=0.99)
model.learn(total_timesteps=10000)

# ========== 4. 测试策略 ==========
def backtest(env, model):
    obs = env.reset()
    done = False
    total_pnl = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_pnl += reward
    
    print(f"Backtest PnL: {total_pnl:.2f}")
    return total_pnl

# 运行回测
backtest(env, model)
