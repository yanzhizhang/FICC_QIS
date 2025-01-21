# order_book.py

import pandas as pd
import numpy as np
from scipy.stats import norm


class FuturesOrderBook:
    def __init__(self, risk_free_rate=0.01):
        """
        Initialize the order book.

        :param risk_free_rate: Risk-free rate used for Sharpe ratio calculation (annualized).
        """
        self.transactions = []  # List to store transactions
        self.risk_free_rate = risk_free_rate  # Annualized risk-free rate

    def record_transaction(self, timestamp, symbol, quantity, price, notional):
        """
        Record a transaction in the order book.

        :param timestamp: The time of the transaction.
        :param symbol: The symbol of the traded future.
        :param quantity: The number of contracts (positive for long, negative for short).
        :param price: The price per contract.
        :param notional: The notional value of the transaction.
        """
        self.transactions.append(
            {"timestamp": pd.to_datetime(timestamp), "symbol": symbol, "quantity": quantity, "price": price, "notional": notional}
        )

    def get_portfolio(self):
        """
        Get the current portfolio based on recorded transactions.

        :return: A DataFrame summarizing the portfolio.
        """
        df = pd.DataFrame(self.transactions)
        if df.empty:
            return pd.DataFrame(columns=["symbol", "net_quantity", "avg_price", "notional"])

        portfolio = (
            df.groupby("symbol")
            .apply(
                lambda x: pd.Series(
                    {
                        "net_quantity": x["quantity"].sum(),
                        "avg_price": (x["quantity"] * x["price"]).sum() / x["quantity"].sum(),
                        "notional": x["notional"].sum(),
                    }
                )
            )
            .reset_index()
        )
        return portfolio

    def calculate_sharpe_ratio(self, returns):
        """
        Calculate the Sharpe ratio for the portfolio.

        :param returns: A Pandas Series of portfolio returns.
        :return: The Sharpe ratio.
        """
        excess_returns = returns - self.risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_var(self, returns, confidence_level=0.95):
        """
        Calculate the Value at Risk (VaR) for the portfolio.

        :param returns: A Pandas Series of portfolio returns.
        :param confidence_level: The confidence level for VaR (default: 0.95).
        :return: The VaR.
        """
        return -np.percentile(returns, (1 - confidence_level) * 100)

    def calculate_leverage(self):
        """
        Calculate the leverage level of the portfolio.

        :return: Leverage level (total notional value / equity).
        """
        df = pd.DataFrame(self.transactions)
        if df.empty:
            return 0

        total_notional = df["notional"].sum()
        equity = df["quantity"].sum() * df["price"].iloc[-1]  # Approximation
        return total_notional / equity if equity != 0 else np.inf

    def analyze_portfolio(self, price_history):
        """
        Analyze the portfolio by calculating Sharpe ratio, VaR, and leverage.

        :param price_history: A Pandas Series of portfolio prices (indexed by date).
        :return: A dictionary with Sharpe ratio, VaR, and leverage level.
        """
        returns = price_history.pct_change().dropna()
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        var = self.calculate_var(returns)
        leverage = self.calculate_leverage()

        return {"sharpe_ratio": sharpe_ratio, "value_at_risk": var, "leverage": leverage}
