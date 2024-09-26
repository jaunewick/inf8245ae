import numpy as np


# Part A:
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Returns the design matrix with an all one column appended

    Args:
        X (np.ndarray): Numpy array of shape [observations, num_features]

    Returns:
        np.ndarray: Numpy array of shape [observations, num_features + 1]
    """

    # Write your code here ...

    return X_bias


# Part B:
def linear_regression_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes $y = Xw$

    Args:
        X (np.ndarray): Numpy array of shape [observations, features]
        w (np.ndarray): Numpy array of shape [features, 1]

    Returns:
        np.ndarray: Numpy array of shape [observations, 1]
    """
    # Write your code here ...
    return y


# Part C:
def linear_regression_optimize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Optimizes MSE fit of $y = Xw$

    Args:
        y (np.ndarray): Numpy array of shape [observations, 1]
        X (np.ndarray): Numpy array of shape [observations, features]

    Returns:
        Numpy array of shape [features, 1]
    """

    # Write your answer here ....

    return w


# Part D
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Evaluate the RMSE between actual and predicted values.

    Parameters:
    y (list or np.array): The actual values.
    y_hat (list or np.array): The predicted values.

    Returns:
    float: The RMSE value.
    """
    # Write your code here ...
    return rmse_err

