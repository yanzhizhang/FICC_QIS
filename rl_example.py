# %%
import pytz
import datetime as dt

from helper.future_price_retriever import FuturePriceRetriever
from helper.spread_data_processor import SpreadDataProcessor

from future_arb.reinforcement_learning.rl import *

# %%


def train():
    # 加载数据
    start_date = "20140601"
    end_date = dt.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d")

    # Example usage
    trading_pair = ["RB", "HC"]
    future_price_retriever = FuturePriceRetriever(start_date=start_date)

    # Retrieve spread data
    rb_hc_day_spread_df = future_price_retriever.get_spread_data(trading_pair, frequency="1d")

    lockback_periods = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    hist_vol_windows = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

    # Process the spread data
    data_processor = SpreadDataProcessor()
    rb_hc_fibo_z_score = data_processor.compute_moving_statistics(rb_hc_day_spread_df, target_col="RB_HC_spread", window=lockback_periods)
    rb_hc_fibo_z_score
    rb_hc_fibonacci_spread_df = data_processor.compute_historical_volatility(
        rb_hc_fibo_z_score, price_cols=["RB_prices", "HC_prices"], window=hist_vol_windows
    )

    # 初始化环境
    env = SpreadTradingEnv(rb_hc_fibonacci_spread_df)
    state_dim = len(env._get_state())
    action_dim = 2

    # 初始化PPO
    agent = PPO(state_dim, action_dim)

    # 训练参数
    episodes = 3000
    max_steps = 5000
    batch_size = 128

    print("开始训练")
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
        agent.update(episode_states, episode_actions, episode_rewards, episode_dones, episode_log_probs)

        # 记录训练过程
        if ep % 50 == 0:
            avg_reward = np.mean(episode_rewards)
            total_return = env.cum_pnl / env.init_balance
            print(f"Episode {ep} | Avg Reward: {avg_reward:.2f} | Total Return: {total_return*100:.1f}%")

    return env, agent, reward, total_return


# %%
env, agent, reward, total_return = train()

# %%
