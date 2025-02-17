# spread_trading_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SpreadTradingEnv(gym.Env):
    def __init__(self, df):
        super(SpreadTradingEnv, self).__init__()

        self.df = df
        self.current_step = 0

        # Define action space: continuous position size (-1 to 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Define state space (normalized inputs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.df.columns),), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.df.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1

        if self.current_step >= len(self.df):
            return self.df.iloc[-1].values, 0, True, {}

        # Get the new price and calculate PnL
        prev_price = self.df.iloc[self.current_step - 1]["RB_HC_spread"]
        new_price = self.df.iloc[self.current_step]["RB_HC_spread"]

        # Position size is determined by action
        position_size = action[0]
        pnl = position_size * (new_price - prev_price)

        # Apply slippage cost
        slippage_cost = abs(position_size) * 0.0003  # Assume 3bps slippage

        # Reward = PnL - Slippage Cost
        reward = pnl - slippage_cost

        return self.df.iloc[self.current_step].values, reward, False, {}
