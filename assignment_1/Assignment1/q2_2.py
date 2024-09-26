import numpy as np
from typing import List, Tuple


def cross_validation_linear_regression(k_folds: int, hyperparameters: List[float],
                                       X: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:
    """
    Perform k-fold cross-validation to find the best hyperparameter for Ridge Regression.

    Args:
        k_folds (int): Number of folds to use.
        hyperparameters (List[float]): List of floats containing the hyperparameter values to search.
        X (np.ndarray): Numpy array of shape [observations, features].
        y (np.ndarray): Numpy array of shape [observations, 1].

    Returns:
        best_hyperparam (float): Value of the best hyperparameter found.
        best_mean_squared_error (float): Best mean squared error corresponding to the best hyperparameter.
        mean_squared_errors (List[float]): List of mean squared errors for each hyperparameter.
    """

    # Write your code here ...

    return best_hyperparam, best_mean_squared_error, mean_squared_errors.tolist()
