df = merged_df.copy()


# Sort the DataFrame by index for proper processing
df = df.sort_index()

# Step 1: Calculate Adjusted Close Prices
df["Adj_Close"] = df["S_DQ_CLOSE"] * df["S_DQ_ADJFACTOR"]

# Step 2: Split into TQQQ and TMF data
tqqq_data = df.xs("TQQQ", level="TICKER")
tmf_data = df.xs("TMF", level="TICKER")

# Step 3: Create daily portfolio value DataFrame
portfolio_daily = pd.DataFrame(index=tqqq_data.index)
portfolio_daily["TQQQ_Adj_Close"] = tqqq_data["Adj_Close"]
portfolio_daily["TMF_Adj_Close"] = tmf_data["Adj_Close"]

# Step 4: Define rebalancing parameters
rebalance_window = 20  # Customize the rebalance window (e.g., every 20 days)
all_dates = portfolio_daily.index.get_level_values("TRADE_DT").unique()
start_date = all_dates.min()
end_date = all_dates.max()
rebalance_dates = calculate_rebalance_dates(start_date, end_date, rebalance_window, all_dates)

# Step 5: Initialize portfolio
initial_value = 100000
weights = {"TQQQ": 0.6, "TMF": 0.4}
shares = {"TQQQ": 0, "TMF": 0}

# Calculate portfolio values
portfolio_daily["Realized_PnL"] = 0.0
portfolio_daily["Unrealized_PnL"] = 0.0
last_rebalance_total = initial_value

for i, date in enumerate(portfolio_daily.index):
    if date in rebalance_dates:
        # Rebalance portfolio
        if i == 0:
            # Initial portfolio setup
            shares["TQQQ"] = (initial_value * weights["TQQQ"]) / portfolio_daily.loc[date, "TQQQ_Adj_Close"]
            shares["TMF"] = (initial_value * weights["TMF"]) / portfolio_daily.loc[date, "TMF_Adj_Close"]

        else:
            # Calculate current total value before rebalancing
            current_tqqq_value = shares["TQQQ"] * portfolio_daily.loc[date, "TQQQ_Adj_Close"]
            current_tmf_value = shares["TMF"] * portfolio_daily.loc[date, "TMF_Adj_Close"]
            current_total = current_tqqq_value + current_tmf_value

            # Update shares based on new weights
            shares["TQQQ"] = (current_total * weights["TQQQ"]) / portfolio_daily.loc[date, "TQQQ_Adj_Close"]
            shares["TMF"] = (current_total * weights["TMF"]) / portfolio_daily.loc[date, "TMF_Adj_Close"]

    # Update daily portfolio values
    portfolio_daily.loc[date, "TQQQ_value"] = shares["TQQQ"] * portfolio_daily.loc[date, "TQQQ_Adj_Close"]
    portfolio_daily.loc[date, "TMF_value"] = shares["TMF"] * portfolio_daily.loc[date, "TMF_Adj_Close"]
    portfolio_daily.loc[date, "Total_value"] = portfolio_daily.loc[date, "TQQQ_value"] + portfolio_daily.loc[date, "TMF_value"]


# Plot NAV
plt.figure(figsize=(12, 6))
plt.plot(
    portfolio_daily.index,
    portfolio_daily["Total_value"],
    label="Portfolio NAV",
    color="blue",
)
plt.title(f"Portfolio NAV (Rebalance every {rebalance_window} days)")
plt.xlabel("Date")
plt.ylabel("NAV")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
