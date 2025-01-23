import numpy as np


class LightGbmLossFunction:
    def __init__(self):
        """
        Initializes the LightGbmLossFunction class.
        """
        pass
    

    def pnl_loss(y_true, y_pred):
        """
        Custom loss function for PnL optimization.

        Parameters:
        - y_true: Actual future price changes.
        - y_pred: Predicted trading signals.

        Returns:
        - Gradient and Hessian for LightGBM optimization.
        """
        # Simulated PnL: Signal * Future Price Change
        pnl = y_pred * y_true

        # Gradient: The derivative of negative PnL with respect to y_pred
        grad = -y_true

        # Hessian: Constant second derivative (simplification for PnL)
        hess = np.ones_like(y_true)

        return grad, hess

    def pnl_loss_with_costs(y_true, y_pred, transaction_cost_rate=0.0003):
        """
        Custom loss function for PnL with transaction costs.

        Parameters:
        - y_true: Actual future price changes.
        - y_pred: Predicted trading signals.
        - transaction_cost_rate: Cost per trade (e.g., 0.03%).

        Returns:
        - Gradient and Hessian for LightGBM optimization.
        """
        # Simulated PnL with transaction costs
        pnl = y_pred * y_true - transaction_cost_rate * np.abs(y_pred)

        # Gradient: Derivative of negative PnL
        grad = -y_true + transaction_cost_rate * np.sign(y_pred)

        # Hessian: Second derivative (simplified)
        hess = np.ones_like(y_true)

        return grad, hess

    def pnl_loss_with_drawdown_penalty(y_true, y_pred, penalty_factor=1.0):
        pnl = y_pred * y_true
        drawdown_penalty = penalty_factor * np.minimum(0, pnl) ** 2  # Penalize negative PnL
        grad = -y_true + 2 * penalty_factor * np.minimum(0, pnl)
        hess = np.ones_like(y_true)
        return grad, hess
