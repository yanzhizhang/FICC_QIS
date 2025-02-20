# %%
import datetime as dt

import pytz
from matplotlib import pyplot as plt

from future_arb.reinforcement_learning.rl_lstm import *
from helper.future_price_retriever import FuturePriceRetriever
from helper.spread_data_processor import SpreadDataProcessor

# %%


def train():
    # 加载数据
    start_date = "20140601"
    end_date = dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d")

    # 获取价差数据
    trading_pair = ["RB", "HC"]
    future_price_retriever = FuturePriceRetriever(start_date=start_date)
    rb_hc_day_spread_df = future_price_retriever.get_spread_data(trading_pair, frequency="1d")

    # 计算技术指标
    lockback_periods = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    hist_vol_windows = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

    data_processor = SpreadDataProcessor()
    rb_hc_fibo_z_score = data_processor.compute_moving_statistics(rb_hc_day_spread_df, target_col="RB_HC_spread", window=lockback_periods)
    rb_hc_fibonacci_spread_df = data_processor.compute_historical_volatility(
        rb_hc_fibo_z_score, price_cols=["RB_prices", "HC_prices"], window=hist_vol_windows
    )

    # 初始化环境
    env = SpreadTradingEnv(
        rb_hc_fibonacci_spread_df,
        init_balance=1e8,  # 初始资金
        contract_size=10,  # 每手10吨
        min_lots=1,  # 最小交易1手
        lookback_window=371,
        transaction_cost=5,  # 每手5元手续费
        slippage=1,  # 每手1元滑点
    )
    state_dim = len(env._get_state())
    action_dim = 2
    hidden_dim = 128

    # 初始化LSTM-PPO
    agent = PPO_LSTM(state_dim, action_dim, hidden_dim)

    # 训练参数
    episodes = 2000
    max_steps = 5000
    batch_size = 128
    seq_length = 120  # LSTM序列长度

    # 训练记录
    training_logs = {
        "episode": [],
        "avg_reward": [],
        "total_return": [],
        "max_drawdown": [],
        "sharpe_ratio": [],
        "highest_return": [],
    }

    print("开始训练")
    for ep in range(episodes):
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_values = []
        episode_log_probs = []

        # 初始化LSTM隐藏状态
        h = torch.zeros(1, 1, agent.policy.lstm.hidden_size)
        c = torch.zeros(1, 1, agent.policy.lstm.hidden_size)
        hidden = (h, c)

        for step in range(max_steps):
            # 将状态转换为序列格式
            if len(episode_states) >= seq_length:
                state_seq = np.array(episode_states[-seq_length:])
            else:
                state_seq = np.array([state] * seq_length)

            # 选择动作
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)  # 添加batch维度
            action_mean, value, hidden = agent.policy(state_tensor, hidden)
            dist = Normal(action_mean, 0.1)  # 添加探索噪声
            action = dist.sample().numpy()[0]  # 去掉batch维度
            log_prob = dist.log_prob(torch.FloatTensor(action)).sum()

            # 执行动作
            next_state, reward, done, highest_return, _ = env.step(action)

            # 存储经验
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_values.append(value.item())
            episode_log_probs.append(log_prob.item())

            state = next_state

            if done:
                break

        # 计算性能指标
        returns = np.array(episode_rewards)
        avg_reward = np.mean(returns)
        total_return = (env.portfolio_value - env.init_balance) / env.init_balance
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

        # 更新策略
        if len(episode_states) >= batch_size:
            agent.update(episode_states, episode_actions, episode_rewards, episode_dones)

        # 记录训练过程
        training_logs["episode"].append(ep)
        training_logs["avg_reward"].append(avg_reward)
        training_logs["total_return"].append(total_return)
        training_logs["max_drawdown"].append(env.max_drawdown)
        training_logs["sharpe_ratio"].append(sharpe_ratio)

        training_logs["highest_return"].append(highest_return)  # for debugging

        # 绘制持仓信息
        if ep % 50 == 0 or ep == episodes - 1:
            rb_positions = [state[-7] * 100 for state in episode_states]  # 恢复原始持仓信息
            hc_positions = [state[-6] * 100 for state in episode_states]  # 恢复原始持仓信息
            print(rb_positions)
            # plt.figure(figsize=(15, 5))
            # plt.plot(rb_positions, label="RB Position")
            # plt.plot(hc_positions, label="HC Position")
            # plt.title(f"Episode {ep} - RB and HC Positions")
            # plt.xlabel("Step")
            # plt.ylabel("Position")
            # plt.legend()
            # plt.show()

        # 打印训练进度
        if ep % 50 == 0 or ep == episodes - 1:
            print(
                f"Episode {ep} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Total Return: {total_return*100:.1f}% | "
                f"Max Drawdown: {env.max_drawdown:.1f} | "
                f"highest_return: {highest_return*100:.1f}% | "
                f"Sharpe Ratio: {sharpe_ratio:.2f}"
            )

    # 训练结果可视化
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


# %%
env, agent, training_logs = train()
# %%
