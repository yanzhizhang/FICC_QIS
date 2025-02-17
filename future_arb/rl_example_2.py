import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS

# ======================
# 环境设置 (更新版)
# ======================
class SpreadTradingEnv:
    def __init__(self, data, init_balance=1e6, min_holding_days=3, lookback_window=377):
        self.data = data
        self.feature_columns = data.columns.tolist()
        self.scaler = StandardScaler()
        self.current_step = lookback_window  # 保证所有滚动指标有效
        self.init_balance = init_balance
        self.min_holding_days = min_holding_days
        self.lookback_window = lookback_window
        
        # 状态参数
        self.position = 0.0  # [-1, 1]
        self.hedge_ratio = 1.0  # 初始对冲比例
        self.holding_days = 0
        self.cum_pnl = 0.0
        self.max_drawdown = 0.0
        self.portfolio_value = init_balance
        
        # 预处理数据
        self._preprocess_data()

    def _preprocess_data(self):
        """数据标准化处理"""
        price_cols = ['HC_prices', 'RB_prices']
        spread_cols = [c for c in self.feature_columns if 'spread' in c]
        zscore_cols = [c for c in self.feature_columns if 'z_score' in c]
        vol_cols = [c for c in self.feature_columns if 'HIST_VOL' in c]

        # 价格数据使用滚动标准化
        for col in price_cols:
            self.data[f'{col}_norm'] = (self.data[col] - self.data[col].rolling(500).mean()) / (self.data[col].rolling(500).std() + 1e-6)
        
        # 其他特征整体标准化
        self.data[zscore_cols + vol_cols] = self.scaler.fit_transform(self.data[zscore_cols + vol_cols])

    def _update_hedge_ratio(self):
        """动态更新对冲比例（协整关系）"""
        window_data = self.data.iloc[self.current_step-self.lookback_window:self.current_step]
        model = OLS(window_data['RB_prices'], window_data['HC_prices'])
        self.hedge_ratio = model.fit().params[0]

    def reset(self):
        self.current_step = self.lookback_window
        self.position = 0.0
        self.hedge_ratio = 1.0
        self.holding_days = 0
        self.cum_pnl = 0.0
        self.max_drawdown = 0.0
        self.portfolio_value = self.init_balance
        return self._get_state()

    def _get_state(self):
        """构建完整状态向量"""
        current_data = self.data.iloc[self.current_step]
        state = [
            # 标准化后的价格
            current_data['HC_prices_norm'],
            current_data['RB_prices_norm'],
            current_data['RB_HC_spread'],
            
            # Z-score特征
            *[current_data[f'z_score_{n}d'] for n in [2,3,5,8,13,21,34,55,89,144,233,377]],
            
            # 历史波动率
            *[current_data[f'HIST_VOL_{n}_RB_prices'] for n in [2,3,5,8,13,21,34,55,89,144,233,377]],
            *[current_data[f'HIST_VOL_{n}_HC_prices'] for n in [2,3,5,8,13,21,34,55,89,144,233,377]],
            
            # 持仓与风险
            self.position,
            self.cum_pnl / self.init_balance,
            self.holding_days / 30.0,  # 归一化
            self.max_drawdown / self.init_balance,
            
            # 市场状态
            self.hedge_ratio
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        delta_pos, new_hedge_ratio = action
        prev_position = self.position
        prev_value = self.portfolio_value
        
        # 更新对冲比例（限制变化幅度）
        self.hedge_ratio = np.clip(new_hedge_ratio, 0.5, 2.0)
        
        # 计算实际价差收益率
        spread_return = (
            (self.data.iloc[self.current_step]['RB_prices'] - 
            self.hedge_ratio * self.data.iloc[self.current_step]['HC_prices']
        ) / (self.data.iloc[self.current_step-1]['RB_prices'] - 
          self.hedge_ratio * self.data.iloc[self.current_step-1]['HC_prices'] + 1e-6 ))
        
        # 更新仓位
        new_position = np.clip(prev_position + delta_pos * 0.1, -1.0, 1.0)  # 限制每次调整幅度
        
        # 持仓天数约束
        if self.holding_days < self.min_holding_days and new_position * prev_position < 0:
            new_position = prev_position
        
        # 计算收益
        pnl = prev_position * self.init_balance * spread_return
        self.cum_pnl += pnl
        self.portfolio_value += pnl
        
        # 更新最大回撤
        current_drawdown = (prev_value - self.portfolio_value) / prev_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # 移动到下一步
        self.current_step += 1
        self.position = new_position
        self.holding_days = self.holding_days + 1 if abs(new_position) > 0.01 else 0
        
        # 终止条件
        done = self.current_step >= len(self.data) - 1 or self.portfolio_value < self.init_balance * 0.8  # 20%亏损终止
        
        # 奖励函数
        reward = self._calculate_reward(pnl, prev_position, new_position)
        
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, pnl, old_pos, new_pos):
        # 交易成本（双边0.1%）
        transaction_cost = abs(new_pos - old_pos) * 0.002 * self.init_balance
        
        # 风险调整收益
        vol = self.data.iloc[self.current_step]['HIST_VOL_21_RB_prices']  # 使用21日波动率
        risk_adj_return = pnl / (vol + 1e-6)
        
        # 回撤惩罚
        drawdown_penalty = 0.2 * self.max_drawdown
        
        # 持仓时间奖励
        holding_bonus = 0.01 * np.log(1 + self.holding_days) if self.holding_days >= self.min_holding_days else 0
        
        return risk_adj_return - transaction_cost - drawdown_penalty + holding_bonus

# ======================
# 改进的PPO模型（处理高维状态）
# ======================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # Actor分支
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # 输出范围[-1,1]
        )
        
        # Critic分支
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        features = self.feature_extractor(state)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, value

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=1e-4, weight_decay=1e-5)
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
    def update(self, states, actions, rewards, dones):
        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        
        # 计算GAE优势
        with torch.no_grad():
            _, values = self.policy(states)
            values = values.squeeze().numpy()
        
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1 or dones[t]:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t+1] - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换旧值
        old_values = torch.FloatTensor(values)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + old_values
        
        # 多次更新
        for _ in range(4):
            # 重新计算新策略的值和分布
            action_means, new_values = self.policy(states)
            dist = Normal(action_means, 1.0)
            log_probs = dist.log_prob(actions).sum(1)
            
            # 重要性采样比率
            ratios = torch.exp(log_probs - old_log_probs)
            
            # PPO裁剪目标
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic损失（Huber损失）
            critic_loss = nn.SmoothL1Loss()(new_values.squeeze(), returns)
            
            # 熵正则化
            entropy = dist.entropy().mean()
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

# ======================
# 训练流程（优化版）
# ======================
def train(data_path):
    # 加载数据
    raw_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 计算额外特征
    raw_data['RB_HC_spread'] = raw_data['RB_prices'] - raw_data['HC_prices']
    
    # 初始化环境
    env = SpreadTradingEnv(raw_data)
    state_dim = len(env._get_state())
    action_dim = 2
    
    # 初始化PPO
    agent = PPO(state_dim, action_dim)
    
    # 训练参数
    episodes = 2000
    max_steps = 5000
    batch_size = 128
    
    for ep in range(episodes):
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_log_probs = []
        
        for step in range(max_steps):
            # 选择动作
            state_tensor = torch.FloatTensor(state)
            action_mean, value = agent.policy(state_tensor)
            dist = Normal(action_mean, 1.0)
            action = dist.sample().numpy()
            log_prob = dist.log_prob(torch.FloatTensor(action)).sum()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_log_probs.append(log_prob.item())
            
            state = next_state
            
            if done:
                break
        
        # 更新策略
        agent.update(episode_states, episode_actions, episode_rewards, episode_dones)
        
        # 记录训练过程
        if ep % 50 == 0:
            avg_reward = np.mean(episode_rewards)
            total_return = env.cum_pnl / env.init_balance
            print(f"Episode {ep} | Avg Reward: {avg_reward:.2f} | Total Return: {total_return*100:.1f}%")

if __name__ == "__main__":
    train("your_spread_data.csv")