# %%
import datetime as dt
import pytz
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.distributions import Normal
from future_arb.reinforcement_learning.rl_lstm import PPO_LSTM, SpreadTradingEnv
from helper.future_price_retriever import FuturePriceRetriever
from helper.spread_data_processor import SpreadDataProcessor

# %%
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

    # Read data from local CSV file
    rb_hc_fibonacci_spread_df = pd.read_csv("rb_hc_fibonacci_spread.csv")

    # Initialize environment
    env = SpreadTradingEnv(
        rb_hc_fibonacci_spread_df,
        init_balance=1e8,  # Initial balance
        contract_size=10,  # Contract size per lot
        min_lots=1,  # Minimum trading lots
        lookback_window=371,
        transaction_cost=10,  # Transaction cost per lot
        slippage=100,  # Slippage per lot
        cost_penalty_ratio=10,
        drawdown_penalty_ratio=10,
    )
    state_dim = len(env._get_state())
    action_dim = 2
    hidden_dim = 32

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


# %%
env, agent, training_logs = train()

# %%


# %%



