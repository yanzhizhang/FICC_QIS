import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS
from statsmodels.tsa.stattools import adfuller
from torch.distributions import Normal

from helper.future_price_retriever import FuturePriceRetriever
from helper.spread_data_processor import SpreadDataProcessor

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpreadTradingEnv:
    def __init__(
        self,
        data,
        init_balance=1e8,  # 初始资金
        contract_size=10,  # 每手合约规模
        max_position=100,  # 最大持仓手数
        min_lots=1,  # 最小交易手数
        lookback_window=400,  # 回溯窗口
        max_drawdown=0.6,  # 最大回撤比例
        transaction_cost=10,  # 每手固定交易成本（元）
        slippage=100,  # 每手固定滑点成本（元）
        normalization_window=90,
        cost_penalty_ratio = 1e-1,
        drawdown_penalty_ratio = 1e-2,
        instrument_pair: Tuple[str, str] = ("RB", "HC"),
    ):
        self.data = data
        self.feature_columns = data.columns.tolist()
        self.scaler = StandardScaler()
        self.current_step = lookback_window
        self.lookback_window = lookback_window
        self.init_balance = init_balance
        self.contract_size = contract_size
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.min_lots = min_lots
        self.cost_penalty_ratio = cost_penalty_ratio
        self.drawdown_penalty_ratio = drawdown_penalty_ratio

        # 交易成本参数
        self.transaction_cost = transaction_cost  # 元/手
        self.slippage = slippage  # 元/手

        # 状态参数
        self.rb_position = 0  # RB持仓手数（正数表示多头）
        self.hc_position = 0  # HC持仓手数（正数表示多头）
        self.hedge_ratio = 1.0  # 对冲比例
        self.holding_days = 0
        self.portfolio_value = init_balance
        self.max_drawdown = 0.0
        self.cash = init_balance

        # 预处理数据（保持原有预处理逻辑）
        self.normalization_window = normalization_window
        self._preprocess_data()

    def _preprocess_data(self):
        """数据标准化处理"""

        # 价格数据使用滚动标准化
        price_cols = ["HC_prices", "RB_prices"]
        for col in price_cols:
            # 使用滞后窗口计算统计量
            rolling_mean = self.data[col].shift(1).rolling(self.normalization_window).mean()
            rolling_std = self.data[col].shift(1).rolling(self.normalization_window).std()
            self.data[f"{col}_norm"] = (self.data[col] - rolling_mean) / (rolling_std + 1e-6)

        # 添加滞后特征
        spread_cols = [c for c in self.feature_columns if "spread" in c]
        zscore_cols = [c for c in self.feature_columns if "z_score" in c]
        vol_cols = [c for c in self.feature_columns if "HIST_VOL" in c]

        # 其他特征整体标准化
        self.data[zscore_cols + spread_cols + vol_cols] = self.scaler.fit_transform(self.data[zscore_cols + spread_cols + vol_cols])

        # 除了 price col， 滞后其他数据一天 avoid look-ahead bias
        temp_price = self.data[price_cols]
        self.data = self.data.shift(1)
        self.data[price_cols] = temp_price

        # 去除 na
        self.data.dropna(inplace=True)

    def _update_hedge_ratio(self):
        """动态更新对冲比例（协整关系）"""
        # window_data = self.data.iloc[self.current_step - self.lookback_window : self.current_step]

        # # Step 1: Regress RB_prices on HC_prices
        # model = OLS(window_data["RB_prices"], window_data["HC_prices"])
        # result = model.fit()
        # hedge_ratio = result.params[0]

        # # Step 2: Test for cointegration using ADF test on residuals
        # residuals = window_data["RB_prices"] - hedge_ratio * window_data["HC_prices"]
        # adf_test = adfuller(residuals)

        # # If the residuals are stationary, update the hedge ratio
        # if adf_test[1] < 0.05:  # p-value < 0.05 indicates stationarity
        #     self.hedge_ratio = hedge_ratio
        # else:
        #     self.hedge_ratio = 1.0  # Default value if not cointegrated

        self.hedge_ratio = 1.0
        
    def reset(self):
        self.current_step = self.lookback_window
        self.rb_position = 0
        self.hc_position = 0
        self.hedge_ratio = 1.0
        self.holding_days = 0
        self.max_drawdown = 0.0
        self.hist_high = self.init_balance
        self.portfolio_value = self.init_balance
        self.cash = self.init_balance
        return self._get_state()

    def _get_state(self):
        """构建状态向量（添加持仓信息）"""
        current_data = self.data.iloc[self.current_step]
        state = [
            # 原始特征
            current_data["HC_prices_norm"],
            current_data["RB_prices_norm"],
            current_data["RB_HC_spread"],
            *[current_data[f"z_score_{n}d"] for n in [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]],
            *[current_data[f"HIST_VOL_{n}_RB_prices"] for n in [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]],
            *[current_data[f"HIST_VOL_{n}_HC_prices"] for n in [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]],
            # 持仓信息（标准化）
            self.rb_position / self.max_position,  # 假设最大持仓 self.max_position 手
            self.hc_position / self.max_position,
            self.hedge_ratio,
            # 账户信息
            (self.portfolio_value - self.init_balance) / self.init_balance,
            self.cash / self.init_balance,
            self.holding_days / 30,  # 假设最大持仓30天
            self.max_drawdown / self.init_balance,
        ]

        return torch.tensor(state, dtype=torch.float32).to(device)

    def _calculate_transaction_cost(self, rb_trade_lots, hc_trade_lots):
        """计算交易成本和滑点"""
        total_lots = abs(rb_trade_lots) + abs(hc_trade_lots)
        return total_lots * (self.transaction_cost + self.slippage)

    def step(self, action):

        # 解析动作：目标RB手数，对冲比例变化量
        target_rb_position, hedge_ratio_change = action
        # 将target_rb_position: [-1,1]映射到[-self.max_position,self.max_position]手
        target_rb_position = int(target_rb_position * self.max_position)

        # 限制对冲比例变化幅度
        self.hedge_ratio = np.clip(self.hedge_ratio + hedge_ratio_change * 0.1, 0.677, 1.5)
        # self.hedge_ratio = self._update_hedge_ratio()

        # 计算目标HC手数（根据对冲比例）
        target_hc_position = -int(target_rb_position * self.hedge_ratio)

        # 计算实际交易手数（考虑最小交易单位）
        rb_trade = target_rb_position - self.rb_position
        hc_trade = target_hc_position - self.hc_position

        # 计算交易成本
        transaction_cost = self._calculate_transaction_cost(rb_trade, hc_trade)

        # 更新持仓和现金
        self.rb_position = target_rb_position
        self.hc_position = target_hc_position
        self.cash -= transaction_cost

        # 计算持仓收益
        current_rb_price = self.data.iloc[self.current_step]["RB_prices"]
        current_hc_price = self.data.iloc[self.current_step]["HC_prices"]
        prev_rb_price = self.data.iloc[self.current_step - 1]["RB_prices"]
        prev_hc_price = self.data.iloc[self.current_step - 1]["HC_prices"]

        rb_pnl = self.rb_position * (current_rb_price - prev_rb_price) * self.contract_size
        hc_pnl = self.hc_position * (current_hc_price - prev_hc_price) * self.contract_size
        total_pnl = rb_pnl + hc_pnl

        # 更新账户
        self.portfolio_value = (
            self.cash + self.rb_position * current_rb_price * self.contract_size + self.hc_position * current_hc_price * self.contract_size
        )

        # 更新历史高点
        self.hist_high = max(self.hist_high, self.portfolio_value)

        # 更新最大回撤
        current_drawdown = self.hist_high - self.portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # 移动到下一步
        self.current_step += 1
        self.holding_days = self.holding_days + 1 if self.rb_position != 0 else 0

        # 终止条件
        done = self.current_step >= len(self.data) - 1 or self.max_drawdown / self.hist_high >= 0.7

        # 奖励函数
        reward = self._calculate_reward(total_pnl, transaction_cost, current_drawdown)

        return self._get_state(), reward, done, self.hist_high / self.init_balance, {}

    def _calculate_reward(self, pnl, transaction_cost, current_drawdown):
        # 风险调整收益
        vol = self.data.iloc[self.current_step]["HIST_VOL_21_RB_prices"]
        risk_adj_return = pnl / (vol * self.contract_size + 1e-6)

        # 成本惩罚
        cost_penalty = transaction_cost * self.cost_penalty_ratio

        # 回撤惩罚
        drawdown_penalty = current_drawdown * self.drawdown_penalty_ratio

        # 持仓集中度奖励
        position_bonus = 0.01 * (abs(self.rb_position) + 0.01 * (abs(self.hc_position)))

        # return risk_adj_return - cost_penalty + position_bonus - drawdown_penalty
        return pnl - cost_penalty + position_bonus - drawdown_penalty


class LSTMActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_size, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(hidden_size, 64), nn.GELU(), nn.Linear(64, action_dim), nn.Tanh())  # 输出范围[-1,1]
        self.critic = nn.Sequential(nn.Linear(hidden_size, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, hidden=None):
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)

        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        action_mean = self.actor(last_out)
        value = self.critic(last_out)
        return action_mean, value, hidden


class PPO_LSTM:
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.policy = LSTMActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)  # Reduced learning rate
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95

    def update(self, batch_states, batch_actions, batch_rewards, batch_dones):
        # 将数据转换为序列格式
        states = torch.FloatTensor(np.array(batch_states))
        actions = torch.FloatTensor(np.array(batch_actions))

        # 初始化LSTM隐藏状态
        h = torch.zeros(1, len(batch_states), self.policy.lstm.hidden_size)
        c = torch.zeros(1, len(batch_states), self.policy.lstm.hidden_size)
        hidden = (h, c)

        # 前向传播获取旧值
        with torch.no_grad():
            _, old_values, _ = self.policy(states.unsqueeze(1), hidden)
            old_values = old_values.squeeze().numpy()

        # 计算GAE优势
        advantages = []
        gae = 0
        for t in reversed(range(len(batch_rewards))):
            if t == len(batch_rewards) - 1:  # 最后一个时间步
                delta = batch_rewards[t] - old_values[t]
            else:
                delta = batch_rewards[t] + self.gamma * (1 - batch_dones[t]) * old_values[t + 1] - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - batch_dones[t]) * gae
            advantages.insert(0, gae)

        # 标准化优势
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多次更新
        for _ in range(10):  # Increased update iterations
            # 获取新策略的输出
            action_means, new_values, _ = self.policy(states.unsqueeze(1), hidden)
            dist = Normal(action_means, 0.1)  # 添加探索噪声
            log_probs = dist.log_prob(actions).sum(1)

            # 计算比率
            ratios = torch.exp(log_probs - log_probs.detach())

            # 计算损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (new_values.squeeze() - (advantages + old_values)).pow(2).mean()

            # 总损失
            loss = actor_loss + critic_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()


def train():
    # Load data
    start_date = "20140601"
    end_date = dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d")

    # Retrieve spread data
    trading_pair = ["RB", "HC"]
    future_price_retriever = FuturePriceRetriever(start_date=start_date)
    rb_hc_day_spread_df = future_price_retriever.get_spread_data(trading_pair, frequency="1d")

    # Calculate technical indicators
    lookback_periods = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    hist_vol_windows = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

    data_processor = SpreadDataProcessor()
    rb_hc_fibo_z_score = data_processor.compute_moving_statistics(rb_hc_day_spread_df, target_col="RB_HC_spread", window=lookback_periods)
    rb_hc_fibonacci_spread_df = data_processor.compute_historical_volatility(
        rb_hc_fibo_z_score, price_cols=["RB_prices", "HC_prices"], window=hist_vol_windows
    )
    
    # Save data to local CSV file
    rb_hc_fibonacci_spread_df.to_csv("rb_hc_fibonacci_spread.csv", index=False)

    # Initialize environment
    env = SpreadTradingEnv(
        rb_hc_fibonacci_spread_df,
        init_balance=1e8,  # Initial balance
        contract_size=10,  # Contract size per lot
        min_lots=1,  # Minimum trading lots
        lookback_window=250,
        transaction_cost=10,  # Transaction cost per lot
        slippage=100,  # Slippage per lot
        cost_penalty_ratio=0,
        drawdown_penalty_ratio=0,
    )
    state_dim = len(env._get_state())
    action_dim = 2
    hidden_dim = 128  # Increased hidden size

    # Initialize LSTM-PPO
    agent = PPO_LSTM(state_dim, action_dim, hidden_dim)

    # Training parameters
    episodes = 2000
    max_steps = 5000
    batch_size = 64
    seq_length = 120  # LSTM sequence length

    # Training logs
    training_logs = {
        "episode": [],
        "avg_reward": [],
        "total_return": [],
        "max_drawdown": [],
        "sharpe_ratio": [],
        "highest_return": [],
    }

    print("Starting training")
    for ep in range(episodes):
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_values = []
        episode_log_probs = []

        # Initialize LSTM hidden state
        h = torch.zeros(1, 1, agent.policy.lstm.hidden_size)
        c = torch.zeros(1, 1, agent.policy.lstm.hidden_size)
        hidden = (h, c)

        for step in range(max_steps):
            # Convert state to sequence format
            state_seq = np.array(episode_states[-seq_length:]) if len(episode_states) >= seq_length else np.array([state] * seq_length)

            # Select action
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)  # Add batch dimension
            action_mean, value, hidden = agent.policy(state_tensor, hidden)
            dist = Normal(action_mean, torch.ones_like(action_mean) * 0.1)  # Add exploration noise
            action = dist.sample().numpy()[0]  # Remove batch dimension
            log_prob = dist.log_prob(torch.FloatTensor(action)).sum()

            # Execute action
            next_state, reward, done, highest_return, _ = env.step(action)

            # Store experience
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_values.append(value.item())
            episode_log_probs.append(log_prob.item())

            state = next_state

            if done:
                break

        # Normalize rewards
        episode_rewards = (episode_rewards - np.mean(episode_rewards)) / (np.std(episode_rewards) + 1e-8)

        # Calculate performance metrics
        returns = np.array(episode_rewards)
        avg_reward = np.mean(returns)
        total_return = (env.portfolio_value - env.init_balance) / env.init_balance
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

        # Update policy
        if len(episode_states) >= batch_size:
            agent.update(episode_states, episode_actions, episode_rewards, episode_dones)

        # Log training process
        training_logs["episode"].append(ep)
        training_logs["avg_reward"].append(avg_reward)
        training_logs["total_return"].append(total_return)
        training_logs["max_drawdown"].append(env.max_drawdown)
        training_logs["sharpe_ratio"].append(sharpe_ratio)
        training_logs["highest_return"].append(highest_return)  # for debugging

        # Plot positions
        if ep % 50 == 0 or ep == episodes - 1:
            rb_positions = [state[-7] * 100 for state in episode_states]  # Restore original positions
            hc_positions = [state[-6] * 100 for state in episode_states]  # Restore original positions
            print(rb_positions)

        # Print training progress
        if ep % 50 == 0 or ep == episodes - 1:
            print(
                f"Episode {ep} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Total Return: {total_return*100:.1f}% | "
                f"Max Drawdown: {env.max_drawdown:.1f} | "
                f"Highest Return: {highest_return*100:.1f}% | "
                f"Sharpe Ratio: {sharpe_ratio:.2f}"
            )

    # Visualize training results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(training_logs["episode"], training_logs["avg_reward"])
    plt.title("Average Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")

    plt.subplot(2, 2, 2)
    plt.plot(training_logs["episode"], np.array(training_logs["total_return"]) * 100)
    plt.title("Total Return (%)")
    plt.xlabel("Episode")
    plt.ylabel("Return (%)")

    plt.subplot(2, 2, 3)
    plt.plot(training_logs["episode"], np.array(training_logs["max_drawdown"]) * 100)
    plt.title("Max Drawdown (%)")
    plt.xlabel("Episode")
    plt.ylabel("Drawdown (%)")

    plt.subplot(2, 2, 4)
    plt.plot(training_logs["episode"], training_logs["sharpe_ratio"])
    plt.title("Sharpe Ratio")
    plt.xlabel("Episode")
    plt.ylabel("Sharpe Ratio")

    plt.tight_layout()
    plt.show()

    return env, agent, training_logs



env, agent, training_logs = train()
