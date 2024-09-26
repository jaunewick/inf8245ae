import numpy as np




def ridge_regression_optimize(y: np.ndarray, X: np.ndarray, hyperparameter: float) -> np.ndarray:
    """Optimizes MSE fit of y = Xw with L2 regularization.

    Args:
        y (np.ndarray): Salary, Numpy array of shape [observations, 1].
        X (np.ndarray): Features (e.g., experience, test_score), Numpy array of shape [observations, features].
        hyperparameter (float): Lambda used in L2 regularization.

    Returns:
        np.ndarray: Optimal parameters (w), Numpy array of shape [features, 1].
    """
    # Write your code here

    return w


