import numpy as np
from typing import List, Tuple

from q2_1 import ridge_regression_optimize
from q1_1 import linear_regression_predict, rmse

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
    fold_size = X.shape[0] / k_folds

    rmse_record = []
    mean_squared_errors_record = np.array([])

    for hyperparameter in hyperparameters:
        for i in range(k_folds):
            start = int(np.round(fold_size * i))
            stop = int(np.round(fold_size * (i+1)))

            X_train = np.concatenate((X[:start], X[stop:]))
            X_val = X[start:stop]

            y_train = np.concatenate((y[:start], y[stop:]))
            y_val = y[start:stop]

            w_star = ridge_regression_optimize(y_train, X_train, hyperparameter)
            y_hat = linear_regression_predict(X_val, w_star)

            rmse_value = rmse(y_val, y_hat)
            rmse_record.append(rmse_value)

        average_rmse = np.mean(rmse_record)
        mean_squared_errors_record = np.append(mean_squared_errors_record, average_rmse)

    best_mean_squared_error_index = np.argmin(mean_squared_errors_record)

    best_hyperparam = hyperparameters[best_mean_squared_error_index]
    best_mean_squared_error = mean_squared_errors_record[best_mean_squared_error_index]
    mean_squared_errors = mean_squared_errors_record

    return best_hyperparam, best_mean_squared_error, mean_squared_errors.tolist()
